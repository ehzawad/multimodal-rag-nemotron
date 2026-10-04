"""Durable at-least-once jobs. PostgreSQL in deployment; SQLite only for CPU tests.

Effects are not exactly-once. Ingestion has immutable IDs; a recovered RAG job can
repeat inference. Fencing prevents an expired worker overwriting a newer result.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Literal

from fastapi import HTTPException
from pydantic import Field, model_validator
from sqlalchemy import (Column, Float, Integer, JSON, MetaData, String, Table, Text,
                        UniqueConstraint, and_, create_engine, func, or_, select, update)
from sqlalchemy.exc import IntegrityError

from .contracts import (Contract, IndexID, IngestRequest, RagRequest, EmbedRequest, RerankRequest,
                        ChatRequest, SearchRequest, digest)


class JobCreate(Contract):
    kind: Literal['rag', 'ingest', 'embed', 'rerank', 'generate', 'search']
    payload: dict

    @model_validator(mode='after')
    def validate_payload(self):
        schemas = {'rag': RagRequest, 'embed': EmbedRequest, 'rerank': RerankRequest,
                   'generate': ChatRequest, 'search': SearchRequest}
        if self.kind in schemas:
            self.payload = schemas[self.kind].model_validate(self.payload).model_dump(exclude_none=True)
        else:
            class Batch(Contract):
                index_id: IndexID
                batch: IngestRequest
            self.payload = Batch.model_validate(self.payload).model_dump(exclude_none=True)
        return self


class EventSubmit(Contract):
    id: str = Field(min_length=1, max_length=128)
    type: Literal['rag.requested', 'documents.ingest', 'embeddings.requested', 'rerank.requested',
                  'generation.requested', 'search.requested']
    data: dict

    @model_validator(mode="after")
    def valid_data(self):
        self.data = self.job().payload
        return self

    def job(self):
        kinds = {'rag.requested': 'rag', 'documents.ingest': 'ingest', 'embeddings.requested': 'embed',
                 'rerank.requested': 'rerank', 'generation.requested': 'generate', 'search.requested': 'search'}
        return JobCreate(kind=kinds[self.type], payload=self.data)


metadata = MetaData()
jobs = Table('jobs', metadata,
    Column('id', String(36), primary_key=True), Column('idempotency_key', String(160), unique=True, nullable=False),
    Column('request_hash', String(64), nullable=False), Column('kind', String(16), nullable=False),
    Column('payload', JSON, nullable=False), Column('status', String(16), nullable=False, index=True),
    Column('attempt', Integer, nullable=False), Column('fence', Integer, nullable=False),
    Column('owner', String(36)), Column('lease_until', Float), Column('available_at', Float, nullable=False),
    Column('created_at', Float, nullable=False), Column('updated_at', Float, nullable=False),
    Column('result', JSON), Column('error', Text), Column('event_seq', Integer, nullable=False))
events = Table('job_events', metadata,
    Column('job_id', String(36), primary_key=True), Column('seq', Integer, primary_key=True),
    Column('type', String(64), nullable=False), Column('data', JSON, nullable=False),
    Column('created_at', Float, nullable=False), UniqueConstraint('job_id', 'seq'))


class Store:
    def __init__(self, url=None):
        self.engine = create_engine(url or os.environ['DATABASE_URL'], pool_pre_ping=True)
        metadata.create_all(self.engine)  # Initial schema v1 only; incompatible upgrades need migrations.
        self.postgres = self.engine.dialect.name == 'postgresql'

    def now(self, conn):
        # A single authoritative database clock in deployed worker coordination.
        if self.postgres:
            return float(conn.execute(select(func.extract('epoch', func.clock_timestamp()))).scalar_one())
        return time.time()

    def event(self, conn, job_id, kind, data, now):
        seq = conn.execute(update(jobs).where(jobs.c.id == job_id)
                           .values(event_seq=jobs.c.event_seq + 1).returning(jobs.c.event_seq)).scalar_one()
        conn.execute(events.insert().values(job_id=job_id, seq=seq, type=kind, data=data, created_at=now))

    @staticmethod
    def public(row):
        keys = ('id', 'kind', 'status', 'attempt', 'created_at', 'updated_at', 'result', 'error')
        return {**{k: row[k] for k in keys}, 'events_url': f"/v1/jobs/{row['id']}/events"}

    def submit(self, request: JobCreate, key: str):
        if not key or len(key) > 160:
            raise HTTPException(422, 'an Idempotency-Key of 1 to 160 characters is required')
        body, identifier = request.model_dump(), str(uuid.uuid4())
        h = digest(body)
        try:
            with self.engine.begin() as conn:
                now = self.now(conn)
                conn.execute(jobs.insert().values(id=identifier, idempotency_key=key, request_hash=h,
                    kind=request.kind, payload=request.payload, status='queued', attempt=0, fence=0,
                    available_at=now, created_at=now, updated_at=now, event_seq=0))
                self.event(conn, identifier, 'job.queued', {}, now)
        except IntegrityError:
            with self.engine.connect() as conn:
                row = conn.execute(select(jobs).where(jobs.c.idempotency_key == key)).mappings().first()
                if row is None:
                    raise
                if row['request_hash'] != h:
                    raise HTTPException(409, 'Idempotency-Key was used for a different request')
                return {**self.public(row), 'replayed': True}
        return {**self.get(identifier), 'replayed': False}

    def get(self, identifier):
        with self.engine.connect() as conn:
            row = conn.execute(select(jobs).where(jobs.c.id == identifier)).mappings().first()
            if row is None:
                raise HTTPException(404, 'job not found')
            return self.public(row)

    def read_events(self, identifier, after=0, limit=100):
        self.get(identifier)
        with self.engine.connect() as conn:
            rows = conn.execute(select(events).where(and_(events.c.job_id == identifier, events.c.seq > after))
                                .order_by(events.c.seq).limit(limit)).mappings().all()
            result = [dict(r) for r in rows]
        return {'events': result, 'next_after': result[-1]['seq'] if result else after}

    def claim(self, owner, lease_seconds=120, max_attempts=3):
        with self.engine.begin() as conn:
            now = self.now(conn)
            eligible = or_(and_(jobs.c.status == 'queued', jobs.c.available_at <= now),
                           and_(jobs.c.status == 'running', jobs.c.lease_until < now))
            query = select(jobs).where(eligible).order_by(jobs.c.available_at, jobs.c.created_at).limit(1)
            if self.postgres:
                query = query.with_for_update(skip_locked=True)
            row = conn.execute(query).mappings().first()
            if row is None:
                return None
            # Also fence the portable/test path; stale claims never win a compare-and-swap.
            condition = and_(jobs.c.id == row['id'], jobs.c.fence == row['fence'], eligible)
            if row['attempt'] >= max_attempts:
                done = conn.execute(update(jobs).where(condition).values(status='failed', updated_at=now,
                                      error='attempt budget exhausted after lease expiry', owner=None, lease_until=None))
                if done.rowcount:
                    self.event(conn, row['id'], 'job.failed', {'reason': 'attempt_budget'}, now)
                return None
            changed = conn.execute(update(jobs).where(condition).values(status='running', owner=owner,
                fence=row['fence'] + 1, attempt=row['attempt'] + 1, lease_until=now + lease_seconds, updated_at=now))
            if changed.rowcount != 1:
                return None
            self.event(conn, row['id'], 'job.started', {'attempt': row['attempt'] + 1,
                       'recovered': row['status'] == 'running'}, now)
            return {**dict(row), 'status': 'running', 'owner': owner, 'fence': row['fence'] + 1,
                    'attempt': row['attempt'] + 1}

    def owned(self, claim, now):
        return and_(jobs.c.id == claim['id'], jobs.c.status == 'running', jobs.c.owner == claim['owner'],
                    jobs.c.fence == claim['fence'], jobs.c.lease_until >= now)

    def heartbeat(self, claim, lease_seconds=120):
        with self.engine.begin() as conn:
            now = self.now(conn)
            return conn.execute(update(jobs).where(self.owned(claim, now))
                                .values(lease_until=now + lease_seconds)).rowcount == 1

    def finish(self, claim, *, result=None, error=None, retryable=False, max_attempts=3):
        with self.engine.begin() as conn:
            now = self.now(conn)
            status = 'succeeded' if error is None else ('queued' if retryable and claim['attempt'] < max_attempts else 'failed')
            changed = conn.execute(update(jobs).where(self.owned(claim, now)).values(status=status,
                result=result, error=error, updated_at=now, owner=None, lease_until=None,
                available_at=now + min(30, 2 ** claim['attempt'])))
            if changed.rowcount != 1:
                return False
            self.event(conn, claim['id'], 'job.retry_scheduled' if status == 'queued' else f'job.{status}',
                       {'attempt': claim['attempt'], 'error': error} if error else {}, now)
            return True

    def ready(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(select(jobs.c.id).limit(1))
            return True
        except Exception:
            return False

    def close(self):
        self.engine.dispose()
