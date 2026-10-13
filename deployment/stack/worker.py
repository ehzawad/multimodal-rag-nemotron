"""A separate process, not a FastAPI background task. Safe restart and bounded retries."""
from __future__ import annotations

import logging
import os
import signal
import threading
import uuid

from .common import HTTP, REQUEST_ID, RemoteError
from .jobs import Store


def execute(store, http, claim):
    stop = threading.Event()
    lost = threading.Event()

    def renew():
        while not stop.wait(15):
            try:
                if not store.heartbeat(claim):
                    lost.set()
                    return
            except Exception:
                lost.set()  # Do not claim ownership after a failed renewal.
                return

    thread = threading.Thread(target=renew, daemon=True)
    thread.start()
    token = REQUEST_ID.set(claim['id'])
    try:
        routes = {
            'rag': ('RAG_URL', 'http://rag:8000', '/v1/rag'),
            'embed': ('EMBED_URL', 'http://embeddings:8000', '/v1/embeddings'),
            'rerank': ('RERANK_URL', 'http://reranking:8000', '/v1/rerank'),
            'generate': ('GEN_URL', 'http://generation:8000', '/v1/chat/completions'),
            'search': ('SEARCH_URL', 'http://retrieval:8000', '/v1/search'),
        }
        if claim['kind'] in routes:
            env, default, path = routes[claim['kind']]
            result = http.call('POST', os.environ.get(env, default) + path, claim['payload'])
        else:
            payload = claim['payload']
            result = http.call('PUT', os.environ.get('SEARCH_URL', 'http://retrieval:8000') +
                               f"/v1/indexes/{payload['index_id']}/documents", payload['batch'], admin=True)
        if not lost.is_set():
            store.finish(claim, result=result)
    except RemoteError as exc:
        # 409 here is an index-writer conflict; immutable-ID conflicts exhaust the finite retry budget.
        retry = exc.status in {409, 429, 500, 502, 503, 504}
        if not lost.is_set():
            store.finish(claim, error=f'upstream HTTP {exc.status}', retryable=retry)
    except Exception:
        logging.exception('worker operation failed; job_id=%s', claim['id'])
        if not lost.is_set():
            store.finish(claim, error='worker operation failed', retryable=False)
    finally:
        stop.set()
        thread.join(timeout=20)
        REQUEST_ID.reset(token)


def main():
    logging.basicConfig(level=logging.INFO)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    store, http, owner = Store(), HTTP(timeout=600), str(uuid.uuid4())
    try:
        while not stop.is_set():
            try:
                claim = store.claim(owner)
                if claim:
                    execute(store, http, claim)
                else:
                    stop.wait(1)
            except Exception:
                logging.exception('worker database/claim failure')
                stop.wait(5)
    finally:
        http.close()
        store.close()


if __name__ == '__main__':
    main()
