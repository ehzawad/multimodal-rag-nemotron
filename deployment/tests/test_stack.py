"""CPU unit/contract tests. Fakes are test-only and make no quality/performance claims."""
import base64
import io
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import update

from stack.api import create_app
from stack.common import BoundaryMiddleware, Gate, HTTP, RemoteError, gpu_slot, rrf
from stack.contracts import (ChatRequest, Document, EmbedRequest, IndexCreate, IngestRequest,
                             Media, RagRequest, RerankRequest, SearchRequest, EMBED_DIM,
                             embedding_identity, digest)
from stack.jobs import EventSubmit, JobCreate, Store, jobs
from stack.models import ModelAPI
from stack.rag import RAG, validated_searches
from stack.retrieval import Retrieval


@pytest.fixture(autouse=True)
def secrets(monkeypatch, tmp_path):
    monkeypatch.setenv('API_KEY', 'test-application-key-12345')
    monkeypatch.setenv('ADMIN_KEY', 'test-admin-key-67890123456')
    monkeypatch.setenv('GPU_LOCK_PATH', str(tmp_path / 'gpu.lock'))


@pytest.fixture
def headers():
    return {'X-API-Key': 'test-application-key-12345', 'X-Admin-Key': 'test-admin-key-67890123456'}


def png():
    buf = io.BytesIO()
    Image.new('RGB', (32, 32)).save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize('payload', [{}, {'text': '  '}, {'image': 'http://169.254.169.254/latest/'},
                                   {'image': '/etc/passwd'}, {'image': 'data:image/png;base64,a'},
                                   {'text': 'a', 'unexpected': True}])
def test_media_rejects_bad_inputs(payload):
    with pytest.raises(ValidationError):
        Media.model_validate(payload)


def test_media_image_and_literal_url_text():
    assert Media(image=png()).model_input()['image'].size == (32, 32)
    assert Media(text='http://example.com').model_input() == {'text': 'http://example.com'}


@pytest.mark.parametrize('kwargs', [{'stream': True}, {'max_tokens': 100000},
                                  {'messages': [{'role': 'user', 'content': [{'type': 'image_url',
                                   'image_url': {'url': 5}}]}]}])
def test_chat_rejects_unsupported_contract(kwargs):
    body = {'messages': [{'role': 'user', 'content': 'hello'}], **kwargs}
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(body)


def test_bounds_and_identity():
    with pytest.raises(ValidationError):
        SearchRequest(query=Media(text='x'), top_k=10, candidates=5)
    with pytest.raises(ValidationError):
        IndexCreate(index_id='../escape', expected_documents=2)
    with pytest.raises(ValidationError):
        IngestRequest(documents=[Document(id='1', content='x'), Document(id='1', content='x')])
    assert embedding_identity()['fingerprint'] == embedding_identity()['fingerprint']
    assert len(embedding_identity()['fingerprint']) == 64
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})


def test_rrf_dedup_and_order():
    result = rrf([['a', 'b', 'a'], ['b', 'c']])
    assert [r[0] for r in result] == ['b', 'a', 'c']
    assert result[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_bounded_gate_and_release():
    gate = Gate(active=1, waiting=0, timeout=0)
    with gate.enter():
        with pytest.raises(HTTPException) as error:
            with gate.enter():
                pass
        assert error.value.status_code == 429
    with pytest.raises(ValueError):
        with gate.enter():
            raise ValueError()
    with gate.enter():
        pass


def test_gpu_lock_prevents_overlap():
    with gpu_slot(timeout=0.02):
        with pytest.raises(TimeoutError):
            with gpu_slot(timeout=0.02):
                pass
    with gpu_slot(timeout=0.02):
        pass


def test_triton_adapter_batch_splitting_and_tensor_contract():
    batches = []
    def handler(request):
        body = json.loads(request.content)
        inp = body['inputs'][0]
        n = inp['shape'][0]
        batches.append(n)
        assert inp['datatype'] == 'BYTES' and inp['shape'] == [n, 1]
        assert all(json.loads(v)['input_type'] == 'query' for v in inp['data'])
        return httpx.Response(200, json={'outputs': [{'name': 'OUTPUT', 'shape': [n, EMBED_DIM],
                                 'datatype': 'FP32', 'data': [1.0] * (n * EMBED_DIM)}]})
    http = HTTP(transport=httpx.MockTransport(handler))
    response = ModelAPI('embeddings', http).embed(EmbedRequest(inputs=[Media(text='x')] * 17))
    assert batches == [8, 8, 1] and len(response['embeddings']) == 17
    http.close()


def test_bad_triton_shape_rejected():
    http = HTTP(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        'outputs': [{'name': 'OUTPUT', 'shape': [1, 3], 'datatype': 'FP32', 'data': [1, 2, 3]}]})))
    with pytest.raises(RemoteError):
        ModelAPI('embeddings', http).embed(EmbedRequest(inputs=[Media(text='x')]))
    http.close()


def test_rerank_returns_indexes_not_documents():
    http = HTTP(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        'outputs': [{'name': 'OUTPUT', 'shape': [2, 1], 'datatype': 'FP32', 'data': [-2, 8]}]})))
    result = ModelAPI('reranking', http).rerank(RerankRequest(query=Media(text='food'),
                      documents=[Media(text='noise'), Media(text='food')], top_k=1))
    assert result['results'] == [{'index': 1, 'score': 8.0}]
    assert result['score_type'] == 'reranker_logit'
    http.close()


class LexicalFake:
    def __init__(self, docs):
        self.docs = docs
    def search(self, query, k, threshold):
        hits = [d['id'] for d in self.docs if query in d['content']]
        coverage = 1.0 if hits else 0.0
        return (hits[:k] if coverage >= threshold else []), coverage


class FakeNetwork:
    """Models the Qdrant REST envelopes, not Qdrant's durability or numerical implementation."""
    def __init__(self):
        self.points = {}
        self.embeds = 0
        self.wrong_identity = False
    def call(self, method, url, payload=None, **kwargs):
        if url.endswith('/v1/embeddings'):
            self.embeds += 1
            identity = embedding_identity()
            if self.wrong_identity:
                identity['fingerprint'] = 'wrong'
            return {'identity': identity, 'embeddings': [[1.0] * EMBED_DIM for _ in payload['inputs']]}
        if url.endswith('/points/count'):
            return {'result': {'count': len(self.points)}}
        if url.endswith('/points/scroll'):
            return {'result': {'points': list(self.points.values()), 'next_page_offset': None}}
        if url.endswith('/points/query'):
            assert payload['params'] == {'exact': True}
            return {'result': {'points': list(self.points.values())[:payload['limit']]}}
        if method == 'POST' and url.endswith('/points'):
            return {'result': [self.points[i] for i in payload['ids'] if i in self.points]}
        if method == 'PUT' and '/points?' in url:
            self.points.update({p['id']: p for p in payload['points']})
            return {'result': {'status': 'completed'}}
        if method == 'PUT' and '/collections/' in url:
            return {'result': True}
        if url.endswith('/collections'):
            return {'result': {'collections': []}}
        raise AssertionError((method, url, payload))


@pytest.fixture
def retrieval(tmp_path):
    network = FakeNetwork()
    service = Retrieval(network, root=tmp_path, lexical_factory=LexicalFake)
    service.create(IndexCreate(index_id='demo', expected_documents=2))
    return service, network


def test_ingestion_immutable_idempotent_publish_and_restart(retrieval, tmp_path):
    service, network = retrieval
    docs = IngestRequest(documents=[Document(id='1', content='banana', image=png()),
                                   Document(id='2', content='spinach')])
    assert service.ingest('demo', docs)['written'] == 2
    assert service.ingest('demo', docs)['replayed'] == 2
    assert network.embeds == 1
    with pytest.raises(HTTPException) as err:
        service.search(SearchRequest(index_id='demo', query=Media(text='banana')))
    assert err.value.status_code == 409
    service.publish('demo')
    assert service.ingest('demo', docs)['replayed'] == 2
    restarted = Retrieval(network, root=tmp_path, lexical_factory=LexicalFake)
    result = restarted.search(SearchRequest(index_id='demo', query=Media(text='banana'), include_images=True))
    assert result['results'][0]['id'] == '1' and result['results'][0]['image'] == png()
    with pytest.raises(HTTPException):
        service.ingest('demo', IngestRequest(documents=[Document(id='3', content='new')]))
    with pytest.raises(HTTPException):
        service.ingest('demo', IngestRequest(documents=[Document(id='1', content='changed')]))


def test_publish_incomplete_and_fingerprint_mismatch(retrieval):
    service, network = retrieval
    with pytest.raises(HTTPException):
        service.publish('demo')
    network.wrong_identity = True
    with pytest.raises(HTTPException) as error:
        service.ingest('demo', IngestRequest(documents=[Document(id='1', content='x')]))
    assert error.value.status_code == 409 and not network.points


def test_index_create_different_spec_conflict(retrieval):
    service, _ = retrieval
    service.create(IndexCreate(index_id='demo', expected_documents=2))
    with pytest.raises(HTTPException):
        service.create(IndexCreate(index_id='demo', expected_documents=3))


def test_snapshot_checksum_fail_closed(retrieval, tmp_path):
    service, net = retrieval
    service.ingest('demo', IngestRequest(documents=[Document(id='1', content='x'), Document(id='2', content='y')]))
    service.publish('demo')
    (tmp_path / 'demo/documents.json').write_text('[]')
    fresh = Retrieval(net, root=tmp_path, lexical_factory=LexicalFake)
    with pytest.raises(HTTPException) as e:
        fresh.snapshot('demo')
    assert e.value.status_code == 503


def test_agent_allowlist_and_budget():
    calls = [{'function': {'name': 'delete_files', 'arguments': '{"query":"bad"}'}},
             {'function': {'name': 'search_recipes', 'arguments': 'not JSON'}}]
    calls += [{'function': {'name': 'search_recipes', 'arguments': json.dumps({'query': f'q{i}'})}} for i in range(5)]
    assert validated_searches({'tool_calls': calls}) == ['q0', 'q1', 'q2']


class RAGNetwork:
    def __init__(self):
        self.searches = []
    def call(self, method, url, payload=None, **kwargs):
        if url.endswith('/v1/search'):
            self.searches.append(payload['query'])
            return {'embedding': embedding_identity(), 'results': [
                {'id': 'a', 'title': 'A', 'content': 'recipe A', 'metadata': {}, 'score': 0.1},
                {'id': 'b', 'title': 'B', 'content': 'recipe B', 'metadata': {}, 'score': 0.09}]}
        if url.endswith('/v1/rerank'):
            return {'results': [{'index': 1, 'score': 9}, {'index': 0, 'score': 2}], 'score_type': 'reranker_logit'}
        if url.endswith('/v1/chat/completions'):
            if payload.get('tools'):
                message = {'tool_calls': [{'function': {'name': 'search_recipes', 'arguments': '{"query":"translated"}'}}]}
            else:
                message = {'content': 'recipe B [1]'}
            return {'choices': [{'message': message}], 'usage': {'completion_tokens': 5}}
        return {'status': 'ready'}


def test_full_rag_citation_order_and_original_query_preserved():
    net = RAGNetwork()
    service = RAG(net)
    events = []
    out = service.query(RagRequest(index_id='demo', query=Media(text='original'), strategy='agent'),
                        progress=lambda t, d: events.append(t))
    assert net.searches == [{'text': 'original'}, {'text': 'translated'}]
    assert out['results'][0]['id'] == 'b' and out['citation_ids']['1'] == 'b'
    assert 'answer_s' in out['timings'] and out['answer'] == 'recipe B [1]'
    assert events == ['planning.completed', 'retrieval.completed', 'reranking.completed', 'generation.completed']


def test_direct_retrieval_no_generation():
    out = RAG(RAGNetwork()).query(RagRequest(query=Media(text='food'), answer=False))
    assert out['strategy'] == 'direct' and out['answer'] is None and out['usage'] == {}


@pytest.fixture
def store(tmp_path):
    s = Store('sqlite:///' + str(tmp_path / 'jobs.db'))
    yield s
    s.close()


def example_job():
    return JobCreate(kind='rag', payload={'query': {'text': 'food'}, 'answer': False})


def test_job_idempotency_conflict_and_event_sequence(store):
    first = store.submit(example_job(), 'test')
    assert store.submit(example_job(), 'test')['id'] == first['id']
    other = JobCreate(kind='rag', payload={'query': {'text': 'changed'}})
    with pytest.raises(HTTPException):
        store.submit(other, 'test')
    claim = store.claim('worker')
    assert claim['id'] == first['id'] and store.heartbeat(claim)
    assert store.finish(claim, result={'ok': True})
    page = store.read_events(first['id'])
    assert [e['seq'] for e in page['events']] == [1, 2, 3]
    assert [e['type'] for e in page['events']] == ['job.queued', 'job.started', 'job.succeeded']
    assert store.read_events(first['id'], after=2)['events'][0]['type'] == 'job.succeeded'
    assert not store.finish(claim, result={'stale': True})


def expire(store, claim):
    with store.engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == claim['id']).values(lease_until=0))


def test_job_lease_reclaim_fences_old_worker(store):
    job = store.submit(example_job(), 'test')
    old = store.claim('old')
    expire(store, old)
    new = store.claim('new')
    assert new['fence'] == old['fence'] + 1
    assert not store.heartbeat(old)
    assert not store.finish(old, result={'bad': True})
    assert store.finish(new, result={'good': True})
    assert store.get(job['id'])['result'] == {'good': True}


def test_crashed_jobs_do_not_stick_after_retry_budget(store):
    job = store.submit(example_job(), 'test')
    for i in range(3):
        claim = store.claim(str(i))
        expire(store, claim)
    assert store.claim('fourth') is None
    assert store.get(job['id'])['status'] == 'failed'


def test_retry_backoff_then_success(store):
    job = store.submit(example_job(), 'test')
    claim = store.claim('worker')
    assert store.finish(claim, error='upstream HTTP 503', retryable=True)
    assert store.claim('worker') is None
    with store.engine.begin() as conn:
        conn.execute(update(jobs).where(jobs.c.id == job['id']).values(available_at=0))
    retry = store.claim('worker')
    assert retry['attempt'] == 2
    store.finish(retry, result={'ok': True})
    assert store.get(job['id'])['status'] == 'succeeded'


def test_job_api_auth_invalid_events_and_sse_replay(store, headers):
    with TestClient(create_app('jobs', service=store)) as client:
        assert client.get('/healthz').status_code == 401
        assert client.get('/readyz', headers=headers).status_code == 200
        assert client.post('/v1/events', headers=headers, json={'id': 'bad', 'type': 'rag.requested',
                                                               'data': {}}).status_code == 422
        key = {**headers, 'Idempotency-Key': 'abc'}
        response = client.post('/v1/jobs', headers=key, json=example_job().model_dump())
        assert response.status_code == 202
        job = response.json()
        assert client.post('/v1/jobs', headers=key, json=example_job().model_dump()).json()['id'] == job['id']
        claim = store.claim('worker')
        store.finish(claim, result={'ok': True})
        with client.stream('GET', f"/v1/jobs/{job['id']}/events/stream", headers={**headers, 'Last-Event-ID': '1'}) as r:
            text = ''.join(r.iter_text())
        assert 'id: 1\n' not in text and 'id: 2\n' in text and 'job.succeeded' in text
        assert client.get(f"/v1/jobs/{job['id']}/events/stream", headers={**headers, 'Last-Event-ID': '-5'}).status_code == 422


def test_ingest_event_requires_admin(store, headers):
    with TestClient(create_app('jobs', service=store)) as client:
        body = {'id': 'batch', 'type': 'documents.ingest', 'data': {'index_id': 'demo',
                'batch': {'documents': [{'id': '1', 'content': 'hello'}]}}}
        assert client.post('/v1/events', headers={'X-API-Key': headers['X-API-Key']}, json=body).status_code == 403
        assert client.post('/v1/events', headers=headers, json=body).status_code == 202


def test_api_validation_redacts_image_and_propagates_id(headers):
    class Dummy:
        def ready(self): return False
    with TestClient(create_app('embeddings', service=Dummy())) as client:
        assert client.get('/readyz', headers=headers).status_code == 503
        r = client.post('/v1/embeddings', headers={**headers, 'X-Request-ID': 'test-rid'}, json={
            'inputs': [{'image': 'SECRET_NOT_AN_IMAGE'}]})
        assert r.status_code == 422 and 'SECRET_NOT_AN_IMAGE' not in r.text
        assert r.headers['X-Request-ID'] == 'test-rid'
        schema = client.get('/openapi.json').json()
        assert '/v1/embeddings' in schema['paths']


def test_missing_secrets_fail_closed(monkeypatch):
    monkeypatch.setenv('API_KEY', '')
    with pytest.raises(RuntimeError): create_app('rag')


def test_docker_topology_static():
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'compose.yaml').read_text())
    services = cfg['services']
    assert set(services) == {'triton-embed', 'triton-rerank', 'vllm', 'embeddings', 'reranking',
                            'generation', 'qdrant', 'postgres', 'retrieval', 'rag', 'jobs', 'worker', 'edge', 'storage-init'}
    assert [name for name, val in services.items() if val.get('ports')] == ['edge']
    for name in ['triton-embed', 'triton-rerank', 'generation']:
        assert 'gpu-lock:/gpu-lock' in services[name]['volumes']
    assert '--load-model=embed' in services['triton-embed']['command']
    assert '--load-model=rerank' in services['triton-rerank']['command']
    assert '--max-num-seqs=${VLLM_MAX_NUM_SEQS:-1}' in services['vllm']['command']
    assert not services['edge'].get('depends_on')
    assert services['triton-rerank']['depends_on'] == {'storage-init': {'condition': 'service_completed_successfully'}}


def test_request_deadline_prevents_dispatch():
    from stack.common import request_budget
    http = HTTP(transport=httpx.MockTransport(lambda _: pytest.fail('expired request was dispatched')))
    with request_budget(-1), pytest.raises(RemoteError) as e:
        http.call('POST', 'http://service/test', {})
    assert e.value.status == 504
    http.close()


def test_catalog_pagination(retrieval):
    service, _ = retrieval
    service.ingest('demo', IngestRequest(documents=[Document(id='1', content='x'), Document(id='2', content='y')]))
    service.publish('demo')
    first = service.catalog('demo', 0, 1)
    second = service.catalog('demo', first['next_offset'], 1)
    assert [first['documents'][0]['id'], second['documents'][0]['id']] == ['1', '2']
    assert second['next_offset'] is None
    assert 'content' not in first['documents'][0]


def test_body_limit_before_parsing(headers):
    class Dummy:
        def ready(self): return True
    app = create_app('embeddings', service=Dummy())
    app.add_middleware(BoundaryMiddleware, limit=50)
    with TestClient(app) as client:
        r = client.post('/v1/embeddings', headers=headers, content=b'x'*51)
        assert r.status_code == 413


def test_worker_dispatch_uses_api_and_persists_result(store):
    from stack.worker import execute
    result = store.submit(example_job(), 'worker-test')
    claim = store.claim('worker')
    class HTTPDouble:
        def call(self, method, url, payload, **kwargs):
            assert method == 'POST' and url.endswith('/v1/rag')
            return {'results': [], 'answer': None}
    execute(store, HTTPDouble(), claim)
    assert store.get(result['id'])['status'] == 'succeeded'


def test_http_errors_not_silently_retried():
    seen = []
    def handler(req):
        seen.append(req)
        return httpx.Response(503, json={'secret': 'do not expose'})
    http = HTTP(transport=httpx.MockTransport(handler))
    with pytest.raises(RemoteError) as e:
        http.call('POST', 'http://service/v1/rag', {})
    assert len(seen) == 1 and 'secret' not in str(e.value)
    http.close()


@pytest.mark.parametrize('event_type,payload,kind,path', [
    ('embeddings.requested', {'inputs': [{'text': 'x'}]}, 'embed', '/v1/embeddings'),
    ('rerank.requested', {'query': {'text': 'x'}, 'documents': [{'text': 'y'}]}, 'rerank', '/v1/rerank'),
    ('generation.requested', {'messages': [{'role': 'user', 'content': 'hi'}]}, 'generate', '/v1/chat/completions'),
    ('search.requested', {'query': {'text': 'x'}}, 'search', '/v1/search'),
])
def test_model_events_are_api_jobs_and_worker_routes_them(store, event_type, payload, kind, path):
    from stack.worker import execute
    request = EventSubmit(id='e1', type=event_type, data=payload).job()
    assert request.kind == kind
    job = store.submit(request, 'e1')
    claim = store.claim('worker')
    class Network:
        def call(self, method, url, payload, **kwargs):
            assert method == 'POST' and url.endswith(path)
            return {'ok': True}
    execute(store, Network(), claim)
    assert store.get(job['id'])['result'] == {'ok': True}
