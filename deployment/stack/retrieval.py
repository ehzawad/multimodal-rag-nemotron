"""Immutable, versioned Qdrant indexes with a matching in-memory BM25 projection.

Only this service writes its index volume. Qdrant owns points; the volume owns
manifests and content-addressed images. A published index cannot be mutated.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException

from .common import HTTP, RemoteError, atomic_json, rrf
from .contracts import (Document, EMBED_DIM, IndexCreate, IngestRequest, SearchRequest,
                        digest, embedding_identity)


class Lexical:
    def __init__(self, documents):
        import bm25s
        self.module, self.ids = bm25s, [d['id'] for d in documents]
        self.engine = bm25s.BM25(backend='numpy')
        texts = [re.sub(r'[#*_`>\[\]()|-]+', ' ', d['content']) for d in documents]
        self.engine.index(bm25s.tokenize(texts, stopwords='en', show_progress=False), show_progress=False)

    def search(self, query, k, threshold):
        tokens = self.module.tokenize([query], stopwords='en', show_progress=False, return_ids=False)[0]
        if not tokens:
            return [], 0.0
        coverage = sum(t in self.engine.vocab_dict for t in tokens) / len(tokens)
        if coverage < threshold:
            return [], coverage
        encoded = self.module.tokenize([query], stopwords='en', show_progress=False)
        if not encoded.vocab or not any(encoded.ids):
            return [], coverage
        indices, scores = self.engine.retrieve(encoded, k=min(k, len(self.ids)),
                                               show_progress=False, backend_selection='numpy')
        return [self.ids[int(i)] for i, s in zip(indices[0], scores[0]) if s > 0], coverage


class Retrieval:
    def __init__(self, http: HTTP, *, root=None, lexical_factory=Lexical):
        self.http = http
        self.root = Path(root or os.environ.get('INDEX_DIR', '/data/indexes'))
        self.root.mkdir(parents=True, exist_ok=True)
        self.qdrant = os.environ.get('QDRANT_URL', 'http://qdrant:6333')
        self.embed_url = os.environ.get('EMBED_URL', 'http://embeddings:8000')
        self.cache, self.cache_lock, self.lexical_factory = {}, threading.Lock(), lexical_factory

    def qcall(self, method, suffix, payload=None):
        return self.http.call(method, self.qdrant + suffix, payload,
                              headers={'api-key': os.environ.get('API_KEY', '')}).get('result')

    @staticmethod
    def collection(index_id):
        return 'mmrag_' + index_id

    @staticmethod
    def point_id(index_id, doc_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'mmrag:{index_id}:{doc_id}'))

    def path(self, index_id):
        # Check again even for internal callers which did not go through Pydantic.
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,62}', index_id):
            raise HTTPException(422, 'invalid index ID')
        return self.root / index_id

    @contextmanager
    def mutation(self, index_id):
        path = self.path(index_id)
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'writer.lock').open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HTTPException(409, 'index writer busy; retry after current batch') from exc
            try:
                yield path
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def manifest(self, index_id):
        try:
            value = json.loads((self.path(index_id) / 'manifest.json').read_text())
        except FileNotFoundError as exc:
            raise HTTPException(404, 'index not found') from exc
        if value['embedding']['fingerprint'] != embedding_identity()['fingerprint']:
            raise HTTPException(409, 'index/model fingerprint mismatch; build a new index')
        return value

    def create(self, request: IndexCreate):
        with self.mutation(request.index_id) as path:
            spec = request.model_dump()
            target = path / 'manifest.json'
            if target.exists():
                existing = self.manifest(request.index_id)
                if existing['spec'] != spec:
                    raise HTTPException(409, 'index already exists with a different specification')
                return existing
            # Qdrant collection creation is idempotent after a process crash before the manifest write.
            name = self.collection(request.index_id)
            try:
                self.qcall('PUT', f'/collections/{name}', {'vectors': {'size': EMBED_DIM, 'distance': 'Cosine'}})
            except RemoteError as exc:
                if exc.status != 409:
                    raise
                info = self.qcall('GET', f'/collections/{name}')
                vectors = info['config']['params']['vectors']
                if vectors.get('size') != EMBED_DIM or vectors.get('distance') != 'Cosine':
                    raise HTTPException(409, 'existing Qdrant collection is incompatible') from exc
            value = dict(schema_version=1, spec=spec, state='building', embedding=embedding_identity(),
                         created_at=time.time())
            atomic_json(target, value)
            return value

    def count(self, index_id):
        result = self.qcall('POST', f'/collections/{self.collection(index_id)}/points/count', {'exact': True})
        return int(result['count'])

    def status(self, index_id):
        return {**self.manifest(index_id), 'documents': self.count(index_id)}

    def list_indexes(self):
        return [self.manifest(p.parent.name) for p in sorted(self.root.glob('*/manifest.json'))]

    def check_identity(self, response, manifest):
        if response.get('identity', {}).get('fingerprint') != manifest['embedding']['fingerprint']:
            raise HTTPException(409, 'embedding server does not match the index fingerprint')

    def ingest(self, index_id, request: IngestRequest):
        with self.mutation(index_id) as path:
            manifest = self.manifest(index_id)
            name = self.collection(index_id)
            point_ids = [self.point_id(index_id, d.id) for d in request.documents]
            existing = self.qcall('POST', f'/collections/{name}/points',
                                  {'ids': point_ids, 'with_payload': True, 'with_vector': False})
            hashes = {p['payload']['id']: p['payload']['content_hash'] for p in existing}
            new = []
            for doc in request.documents:
                h = digest(doc.model_dump())
                if doc.id in hashes:
                    if hashes[doc.id] != h:
                        raise HTTPException(409, 'document ID already exists with different content')
                else:
                    new.append(doc)
            # A retry of a fully committed batch is safe even after publication.
            if not new:
                return {'index_id': index_id, 'written': 0, 'replayed': len(request.documents)}
            if manifest['state'] != 'building':
                raise HTTPException(409, 'published indexes are immutable; create a new index version')
            if self.count(index_id) + len(new) > manifest['spec']['expected_documents']:
                raise HTTPException(409, 'batch exceeds declared index size')
            inputs = []
            payloads = []
            for doc in new:
                text = doc.content.split('\n## Reviews')[0] if manifest['spec']['transform'] == 'recipe_v1' else doc.content
                inputs.append({'text': text, **({'image': doc.image} if doc.image else {})})
                payload = doc.model_dump(exclude={'image'})
                payload['content_hash'] = digest(doc.model_dump())
                if doc.image:
                    image_ref = digest(doc.image)
                    atomic_json(path / 'images' / f'{image_ref}.json', doc.image)
                    payload['image_ref'] = image_ref
                payloads.append(payload)
            response = self.http.call('POST', self.embed_url + '/v1/embeddings',
                                       {'input_type': 'document', 'inputs': inputs})
            self.check_identity(response, manifest)
            vectors = response['embeddings']
            if len(vectors) != len(new) or any(len(v) != EMBED_DIM for v in vectors):
                raise RemoteError(502, 'embedding-shape')
            points = [{'id': self.point_id(index_id, d.id), 'vector': v, 'payload': p}
                      for d, v, p in zip(new, vectors, payloads, strict=True)]
            self.qcall('PUT', f'/collections/{name}/points?wait=true', {'points': points})
            return {'index_id': index_id, 'written': len(new), 'replayed': len(request.documents) - len(new)}

    def all_documents(self, index_id):
        offset = None
        documents = []
        while True:
            payload = {'limit': 256, 'with_payload': True, 'with_vector': False}
            if offset is not None:
                payload['offset'] = offset
            page = self.qcall('POST', f'/collections/{self.collection(index_id)}/points/scroll', payload)
            documents.extend(p['payload'] for p in page['points'])
            offset = page.get('next_page_offset')
            if offset is None:
                break
        return sorted(documents, key=lambda d: d['id'])

    def publish(self, index_id):
        with self.mutation(index_id) as path:
            manifest = self.manifest(index_id)
            if manifest['state'] == 'published':
                return manifest
            docs = self.all_documents(index_id)
            if len(docs) != manifest['spec']['expected_documents']:
                raise HTTPException(409, 'cannot publish: document count differs from expected_documents')
            projection = self.lexical_factory(docs)  # Fail before publishing if tokenization/indexing fails.
            atomic_json(path / 'documents.json', docs)
            manifest.update(state='published', published_at=time.time(), documents_hash=digest(docs))
            atomic_json(path / 'manifest.json', manifest)
            with self.cache_lock:
                self.cache[index_id] = (docs, projection)
            return manifest

    def snapshot(self, index_id):
        manifest = self.manifest(index_id)
        if manifest['state'] != 'published':
            raise HTTPException(409, 'index is not published')
        with self.cache_lock:
            if index_id not in self.cache:
                docs = json.loads((self.path(index_id) / 'documents.json').read_text())
                if digest(docs) != manifest['documents_hash']:
                    raise HTTPException(503, 'index snapshot checksum mismatch')
                self.cache[index_id] = (docs, self.lexical_factory(docs))
            return manifest, *self.cache[index_id]

    def catalog(self, index_id, offset=0, limit=1000):
        manifest, docs, _ = self.snapshot(index_id)
        page = [{k: d[k] for k in ('id', 'title', 'metadata')} for d in docs[offset:offset + limit]]
        return {'documents': page, 'next_offset': offset + limit if offset + limit < len(docs) else None,
                'documents_hash': manifest['documents_hash']}

    def ready(self):
        try:
            self.qcall('GET', '/collections')
            return True
        except Exception:
            return False

    def search(self, request: SearchRequest):
        started = time.monotonic()
        manifest, docs, lexical = self.snapshot(request.index_id)
        by_id = {d['id']: d for d in docs}
        rankings, coverage = [], None
        lexical_text = request.lexical_text or request.query.text
        if request.mode != 'lexical':
            inputs = []
            if request.query.text:
                inputs.append({'text': request.query.text})
            if request.query.image:
                inputs.append({'image': request.query.image})
            embedded = self.http.call('POST', self.embed_url + '/v1/embeddings',
                                      {'input_type': 'query', 'inputs': inputs,
                                       'instruction': manifest['spec']['query_instruction']})
            self.check_identity(embedded, manifest)
            for vector in embedded['embeddings']:
                result = self.qcall('POST', f'/collections/{self.collection(request.index_id)}/points/query',
                                    {'query': vector, 'limit': request.candidates,
                                     'params': {'exact': request.exact}, 'with_payload': True})
                rankings.append([p['payload']['id'] for p in result['points']])
        if request.mode != 'dense' and lexical_text:
            ids, coverage = lexical.search(lexical_text, request.candidates,
                                           0 if request.mode == 'lexical' else request.bm25_min_coverage)
            if ids:
                rankings.append(ids)
        results = []
        for doc_id, score in rrf(rankings)[:request.top_k]:
            if doc_id not in by_id:
                raise HTTPException(503, 'Qdrant and index snapshot disagree')
            doc = by_id[doc_id]
            item = {k: doc[k] for k in ('id', 'title', 'content', 'metadata')}
            if request.include_images and doc.get('image_ref'):
                item['image'] = json.loads((self.path(request.index_id) / 'images' / f"{doc['image_ref']}.json").read_text())
            item['score'] = score
            results.append(item)
        return {'index_id': request.index_id, 'embedding': manifest['embedding'], 'score_type': 'rrf',
                'bm25_coverage': coverage, 'results': results, 'elapsed_s': time.monotonic() - started}
