"""Each SERVICE_ROLE starts a different API process/container with its own lifecycle."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool

from .common import BoundaryMiddleware, HTTP, REQUEST_ID, RemoteError
from .contracts import (ChatRequest, EmbedRequest, IndexCreate, IngestRequest, RagRequest,
                        RerankRequest, SearchRequest, embedding_identity)
from .jobs import EventSubmit, JobCreate


def create_app(role=None, service=None):
    role = role or os.environ.get('SERVICE_ROLE', 'rag')
    if role not in {'embeddings', 'reranking', 'generation', 'retrieval', 'rag', 'jobs'}:
        raise ValueError('unknown SERVICE_ROLE')
    key, admin_key = os.environ.get('API_KEY', ''), os.environ.get('ADMIN_KEY', '')
    if len(key) < 16 or len(admin_key) < 16 or key == admin_key:
        raise RuntimeError('distinct API_KEY and ADMIN_KEY secrets, each >=16 characters, are required')

    def authorized(x_api_key: str = Header(default='')):
        if not hmac.compare_digest(x_api_key, key):
            raise HTTPException(401, 'invalid API key')

    def administrator(x_admin_key: str = Header(default='')):
        if not hmac.compare_digest(x_admin_key, admin_key):
            raise HTTPException(403, 'administrative key required')

    @asynccontextmanager
    async def lifespan(app):
        logging.basicConfig(level=logging.INFO)
        http = HTTP()
        try:
            if service is not None:  # Explicit dependency injection for CPU contract tests.
                instance = service
            elif role in {'embeddings', 'reranking', 'generation'}:
                from .models import ModelAPI
                instance = ModelAPI(role, http)
            elif role == 'retrieval':
                from .retrieval import Retrieval
                instance = Retrieval(http)
            elif role == 'rag':
                from .rag import RAG
                instance = RAG(http)
            else:
                from .jobs import Store
                instance = Store()
            app.state.service = instance
            yield
        finally:
            http.close()
            if service is None and role == 'jobs' and hasattr(app.state, 'service'):
                app.state.service.close()

    app = FastAPI(title=f'MMRAG {role}', version='1.0.0', lifespan=lifespan,
                  dependencies=[Depends(authorized)], root_path=f'/services/{role}')
    app.add_middleware(BoundaryMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]}
                                        for e in exc.errors()]}, status_code=422)

    @app.exception_handler(RemoteError)
    async def upstream_error(request, exc):
        status = exc.status if exc.status in {400, 409, 413, 422, 429, 503, 504} else 502
        return JSONResponse({'detail': 'upstream operation failed', 'upstream': exc.service,
                             'request_id': REQUEST_ID.get()}, status_code=status,
                            headers={'Retry-After': '2'} if status == 429 else None)

    @app.get('/healthz')
    def health():
        return {'status': 'alive', 'service': role}

    @app.get('/readyz')
    def ready(request: Request):
        if not request.app.state.service.ready():
            raise HTTPException(503, 'required dependency is not ready')
        return {'status': 'ready', 'service': role}

    @app.get('/v1/info')
    def info():
        return {'service': role, 'api_version': '1.0.0', 'embedding': embedding_identity(),
                'deployment': os.environ.get('DEPLOYMENT_REVISION', 'unrecorded')}

    if role == 'embeddings':
        @app.post('/v1/embeddings')
        def embeddings(body: EmbedRequest, request: Request):
            return request.app.state.service.embed(body)
    elif role == 'reranking':
        @app.post('/v1/rerank')
        def rerank(body: RerankRequest, request: Request):
            return request.app.state.service.rerank(body)
    elif role == 'generation':
        @app.post('/v1/chat/completions')
        def chat(body: ChatRequest, request: Request):
            return request.app.state.service.chat(body)
    elif role == 'retrieval':
        @app.get('/v1/indexes')
        def indexes(request: Request):
            return {'indexes': request.app.state.service.list_indexes()}

        @app.post('/v1/indexes', dependencies=[Depends(administrator)], status_code=201)
        def create_index(body: IndexCreate, request: Request):
            return request.app.state.service.create(body)

        @app.get('/v1/indexes/{index_id}')
        def index_status(index_id: str, request: Request):
            return request.app.state.service.status(index_id)

        @app.get('/v1/indexes/{index_id}/documents')
        def catalog(index_id: str, request: Request, offset: int = Query(0, ge=0),
                    limit: int = Query(1000, ge=1, le=1000)):
            return request.app.state.service.catalog(index_id, offset, limit)

        @app.put('/v1/indexes/{index_id}/documents', dependencies=[Depends(administrator)])
        def ingest(index_id: str, body: IngestRequest, request: Request):
            return request.app.state.service.ingest(index_id, body)

        @app.post('/v1/indexes/{index_id}/publish', dependencies=[Depends(administrator)])
        def publish(index_id: str, request: Request):
            return request.app.state.service.publish(index_id)

        @app.post('/v1/search')
        def search(body: SearchRequest, request: Request):
            return request.app.state.service.search(body)
    elif role == 'rag':
        @app.post('/v1/rag')
        def rag(body: RagRequest, request: Request):
            return request.app.state.service.query(body)
    elif role == 'jobs':
        def submit(body, key, request, admin):
            if body.kind == 'ingest':
                administrator(admin)
            return request.app.state.service.submit(body, key)

        @app.post('/v1/jobs', status_code=202)
        def job(body: JobCreate, request: Request, idempotency_key: str = Header(default=''),
                x_admin_key: str = Header(default='')):
            return submit(body, 'request:' + idempotency_key if idempotency_key else '', request, x_admin_key)

        @app.post('/v1/events', status_code=202)
        def event(body: EventSubmit, request: Request, x_admin_key: str = Header(default='')):
            return submit(body.job(), 'event:' + body.id, request, x_admin_key)

        @app.get('/v1/jobs/{job_id}')
        def get_job(job_id: str, request: Request):
            return request.app.state.service.get(job_id)

        @app.get('/v1/jobs/{job_id}/events')
        def events(job_id: str, request: Request, after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=100)):
            return request.app.state.service.read_events(job_id, after, limit)

        @app.get('/v1/jobs/{job_id}/events/stream')
        async def stream(job_id: str, request: Request, after: int = Query(0, ge=0),
                         last_event_id: str | None = Header(default=None)):
            await run_in_threadpool(request.app.state.service.get, job_id)
            try:
                cursor = int(last_event_id) if last_event_id is not None else after
                if cursor < 0:
                    raise ValueError()
            except ValueError:
                raise HTTPException(422, 'Last-Event-ID must be a nonnegative sequence number')

            async def generate():
                nonlocal cursor
                while not await request.is_disconnected():
                    page = await run_in_threadpool(request.app.state.service.read_events, job_id, cursor, 100)
                    for item in page['events']:
                        cursor = item['seq']
                        yield f"id: {cursor}\nevent: {item['type']}\ndata: {json.dumps(item)}\n\n"
                    state = await run_in_threadpool(request.app.state.service.get, job_id)
                    if state['status'] in {'succeeded', 'failed'} and len(page['events']) < 100:
                        # Recheck once: completion may commit between the page read and status read.
                        tail = await run_in_threadpool(request.app.state.service.read_events, job_id, cursor, 100)
                        if tail['events']:
                            continue
                        return
                    yield ': keepalive\n\n'
                    await asyncio.sleep(1)
            return StreamingResponse(generate(), media_type='text/event-stream',
                                     headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    return app
