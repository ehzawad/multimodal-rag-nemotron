"""Transport, bounded admission, image-safe clients, and same-host GPU coordination."""
from __future__ import annotations

import contextvars
import fcntl
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import httpx
from fastapi import HTTPException
from starlette.responses import JSONResponse

DEADLINE = contextvars.ContextVar("deadline", default=None)
REQUEST_ID = contextvars.ContextVar("request_id", default="")
log = logging.getLogger("mmrag")


class RemoteError(RuntimeError):
    def __init__(self, status: int, service: str):
        self.status = status
        self.service = service
        super().__init__(f"{service} returned HTTP {status}")


class HTTP:
    def __init__(self, *, transport=None, key=None, timeout=180):
        self.timeout = timeout
        self.key = key if key is not None else os.environ.get("API_KEY", "")
        self.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=5, pool=5),
                                  limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                                  transport=transport, follow_redirects=False)

    def call(self, method, url, payload=None, *, admin=False, headers=None):
        h = {"X-API-Key": self.key, "X-Request-ID": REQUEST_ID.get() or str(uuid.uuid4())}
        if admin:
            h["X-Admin-Key"] = os.environ.get("ADMIN_KEY", "")
        h.update(headers or {})
        timeout = self.timeout
        if DEADLINE.get() is not None:
            timeout = min(timeout, DEADLINE.get() - time.monotonic())
            if timeout <= 0:
                raise RemoteError(504, "request-deadline")
        try:
            r = self.client.request(method, url, json=payload, headers=h,
                                    timeout=httpx.Timeout(timeout, connect=min(5, timeout), pool=min(5, timeout)))
        except httpx.TimeoutException as exc:
            raise RemoteError(504, url.split("/")[2]) from exc
        except httpx.HTTPError as exc:
            raise RemoteError(503, url.split("/")[2]) from exc
        if r.status_code >= 400:
            raise RemoteError(r.status_code, url.split("/")[2])
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError as exc:
            raise RemoteError(502, url.split("/")[2]) from exc

    def close(self):
        self.client.close()


@contextmanager
def request_budget(seconds):
    token = DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        DEADLINE.reset(token)


class Gate:
    """A finite in-process queue, not an unbounded Semaphore waiter list."""
    def __init__(self, active=1, waiting=4, timeout=2):
        self.total = threading.BoundedSemaphore(active + waiting)
        self.active = threading.BoundedSemaphore(active)
        self.timeout = timeout

    @contextmanager
    def enter(self):
        if not self.total.acquire(blocking=False):
            raise HTTPException(429, "queue is full", headers={"Retry-After": "2"})
        entered = False
        try:
            entered = self.active.acquire(timeout=self.timeout)
            if not entered:
                raise HTTPException(429, "queue wait expired", headers={"Retry-After": "2"})
            yield
        finally:
            if entered:
                self.active.release()
            self.total.release()


@contextmanager
def gpu_slot(timeout=None):
    """One shared Linux flock across all three model paths on the single-GPU profile.

    This controls overlapping inference, not allocation or cross-host scheduling.
    Kernel releases the lock on process death. Never use a separate lock per container.
    """
    if os.environ.get("GPU_SERIALIZE", "1") == "0":
        yield
        return
    path = Path(os.environ.get("GPU_LOCK_PATH", "/gpu-lock/inference.lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + (timeout if timeout is not None else 5)
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU_BUSY")
                time.sleep(0.025)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, identifier in enumerate(dict.fromkeys(ranking), 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


class BoundaryMiddleware:
    """Limit bodies before JSON/image decoding, propagate IDs, and omit content from logs."""
    def __init__(self, app, limit=40 * 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        headers = dict(scope.get("headers", []))
        rid = headers.get(b"x-request-id", b"").decode("ascii", "ignore")
        if not rid or len(rid) > 128 or not all(c.isalnum() or c in "-_" for c in rid):
            rid = str(uuid.uuid4())
        token = REQUEST_ID.set(rid)
        status = 500
        try:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.limit:
                    status = 413
                    return await JSONResponse({"detail": "request body too large"}, 413)(scope, receive, send)
                if not message.get("more_body", False):
                    break
            sent = False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            async def respond(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                    message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", rid.encode())]
                await send(message)

            await self.app(scope, replay, respond)
        finally:
            log.info(json.dumps(dict(request_id=rid, method=scope["method"], status=status,
                                     elapsed_s=round(time.monotonic() - started, 4))))
            REQUEST_ID.reset(token)
