#!/usr/bin/env python3
"""Machine-side deployment verification and API-only data/evaluation operations.

init-env/doctor use the standard library. seed-recipes additionally uses datasets
and huggingface_hub; install requirements-tools.txt in a host virtual environment.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
ROLES = ('embeddings', 'reranking', 'generation', 'retrieval', 'rag', 'jobs')


def envfile():
    path = HERE / '.env'
    if not path.exists():
        raise RuntimeError('Run python manage.py init-env first')
    for line in path.read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key, value)


def command(args):
    result = subprocess.run(args, cwd=HERE, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed:\n{result.stderr[-4000:]}')
    return result.stdout.strip()


def init_env():
    target = HERE / '.env'
    if target.exists():
        raise RuntimeError('.env already exists; refusing to replace deployment secrets')
    content = (HERE / 'env.example').read_text()
    replacements = {
        'replace-with-a-long-random-application-key': secrets.token_hex(32),
        'replace-with-a-different-long-random-admin-key': secrets.token_hex(32),
        'replace-with-a-random-database-password': secrets.token_hex(24),
    }
    try:
        revision = command(['git', 'rev-parse', 'HEAD'])
    except RuntimeError:
        revision = 'source-archive'
    replacements['record-the-git-commit'] = revision
    for old, new in replacements.items():
        content = content.replace(old, new)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(content)
    print('Created .env with distinct random secrets and mode 0600. No secrets printed.')


def save(name, value):
    directory = HERE / 'artifacts'
    directory.mkdir(exist_ok=True)
    path = directory / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)
    print(f'Saved {path}')


def doctor(containers=False):
    envfile()
    if sys.platform != 'linux':
        raise RuntimeError('GPU deployment targets Linux, not Docker Desktop on macOS')
    result = {'time': time.time(), 'python': sys.version,
              'gpu': command(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used',
                              '--format=csv,noheader']),
              'docker': command(['docker', '--version']),
              'compose': command(['docker', 'compose', 'version']),
              'disk': command(['df', '-h', str(HERE)]),
              'memory': command(['free', '-h'])}
    command(['docker', 'compose', 'config', '--quiet'])  # Never save interpolated secrets.
    if containers:
        probe = ("import json,torch,importlib.metadata as m; "
                 "assert torch.cuda.is_available(); "
                 "x=torch.ones((64,64),device='cuda'); y=x@x; torch.cuda.synchronize(); "
                 "assert y[0,0].item()==64; "
                 "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                 "'device':torch.cuda.get_device_name(0)}))")
        for label, image, gpu in [('triton', 'mmrag-triton:deployment-v1', os.environ.get('EMBED_GPU', '0')),
                                  ('vllm', os.environ['VLLM_IMAGE'], os.environ.get('GEN_GPU', '0'))]:
            result[label] = command(['docker', 'run', '--rm', '--gpus', f'device={gpu}',
                                     '--entrypoint', 'python3', image, '-c', probe])
        # An early CLI check catches removed/renamed deployment flags without downloading weights.
        help_text = command(['docker', 'run', '--rm', '--entrypoint', 'vllm',
                             os.environ['VLLM_IMAGE'], 'serve', '--help=all'])
        for flag in ['--revision', '--gpu-memory-utilization', '--max-model-len', '--limit-mm-per-prompt',
                     '--tool-call-parser', '--default-chat-template-kwargs']:
            if flag not in help_text:
                raise RuntimeError(f'The selected vLLM image does not advertise {flag}')
        result['vllm_cli_flags'] = 'checked'
    save('doctor.json', result)
    print(result['gpu'])
    print('Host checks passed. This does NOT establish model fit, model quality, or production readiness.')


class APIError(RuntimeError):
    def __init__(self, code, text):
        self.code = code
        super().__init__(f'HTTP {code}: {text[:1000]}')


class API:
    def __init__(self, base=None):
        envfile()
        self.base = (base or f"http://127.0.0.1:{os.environ.get('HTTP_PORT', '8080')}").rstrip('/')
        self.headers = {'X-API-Key': os.environ['API_KEY'], 'X-Admin-Key': os.environ['ADMIN_KEY'],
                        'Content-Type': 'application/json'}

    def call(self, method, path, payload=None, *, key=None, retries=0):
        headers = dict(self.headers)
        if key:
            headers['Idempotency-Key'] = key
        for attempt in range(retries + 1):
            req = urllib.request.Request(self.base + path, method=method, headers=headers,
                      data=json.dumps(payload).encode() if payload is not None else None)
            try:
                with urllib.request.urlopen(req, timeout=610) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                text = exc.read().decode(errors='replace')
                if exc.code in {409, 429, 502, 503, 504} and attempt < retries:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise APIError(exc.code, text) from exc
        raise AssertionError('unreachable')

    def wait(self, job_id, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.call('GET', '/v1/jobs/' + job_id)
            if state['status'] == 'failed':
                raise RuntimeError(f"job {job_id} failed: {state['error']}")
            if state['status'] == 'succeeded':
                return state
            time.sleep(1)
        raise TimeoutError(f'Job {job_id} still pending; inspect its status/events before resubmitting')


def image_url(image):
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def fixtures():
    from PIL import Image, ImageDraw
    rows = [
        ('banana', 'Banana bread', 'Banana bread\nIngredients: banana, flour, egg. Bake for 45 minutes.'),
        ('spinach', 'Spinach chickpea stew', 'Vegan spinach chickpea stew\nIngredients: spinach, chickpeas, tomato. Simmer 20 minutes.'),
        ('omelet', 'Egg omelet', 'Egg omelet\nIngredients: eggs, butter, salt. Cook in a skillet for 8 minutes.'),
    ]
    docs = []
    for identifier, title, text in rows:
        im = Image.new('RGB', (384, 256), 'white')
        ImageDraw.Draw(im).text((12, 12), title, fill='black')
        docs.append({'id': identifier, 'title': title, 'content': text, 'image': image_url(im),
                     'metadata': {'fixture': True}})
    return docs


def seed_fixture(api):
    api.call('POST', '/v1/indexes', {'index_id': 'smoke-v1', 'expected_documents': 3,
                                    'source': 'synthetic-protocol-fixtures-v1'})
    body = {'id': 'smoke-fixtures-' + uuid.uuid4().hex, 'type': 'documents.ingest',
            'data': {'index_id': 'smoke-v1', 'batch': {'documents': fixtures()}}}
    first = api.call('POST', '/v1/events', body)
    duplicate = api.call('POST', '/v1/events', body)
    if first['id'] != duplicate['id']:
        raise RuntimeError('event idempotency check failed')
    api.wait(first['id'])
    api.call('POST', '/v1/indexes/smoke-v1/publish')
    return first['id']


def smoke(api):
    evidence = {'time': time.time(), 'checks': {}, 'services': {}}
    for role in ROLES:
        evidence['services'][role] = api.call('GET', f'/services/{role}/v1/info')
        api.call('GET', f'/services/{role}/readyz')
    embedded = api.call('POST', '/v1/embeddings', {'inputs': [{'text': 'banana bread'}] * 2})
    vectors = embedded['embeddings']
    if len(vectors) != 2 or any(len(v) != 2048 for v in vectors):
        raise RuntimeError('bad embedding shape')
    if any(not all(math.isfinite(x) for x in v) or abs(sum(x*x for x in v) - 1) > 0.03 for v in vectors):
        raise RuntimeError('embedding is nonfinite or not unit normalized')
    evidence['checks']['embedding_shape_norm'] = True
    image = fixtures()[0]['image']
    api.call('POST', '/v1/embeddings', {'inputs': [{'image': image}, {'text': 'banana', 'image': image}]})
    ranked = api.call('POST', '/v1/rerank', {'query': {'text': 'banana bread'},
                     'documents': [{'text': 'banana bread'}, {'text': 'spinach stew'}], 'top_k': 2})
    if sorted(r['index'] for r in ranked['results']) != [0, 1]:
        raise RuntimeError('invalid rerank permutation')
    api.call('POST', '/v1/rerank', {'query': {'text': 'banana bread'},
                                  'documents': [{'text': 'banana bread', 'image': image}], 'top_k': 1})
    evidence['checks']['rerank_text_and_image'] = True
    seed_id = seed_fixture(api)
    for mode in ('dense', 'lexical', 'hybrid'):
        result = api.call('POST', '/v1/search', {'index_id': 'smoke-v1', 'query': {'text': 'banana'}, 'mode': mode})
        if not result['results']:
            raise RuntimeError(f'{mode} search returned no fixture results')
    prompt = {'messages': [{'role': 'user', 'content': 'Reply with one short greeting.'}], 'max_tokens': 32}
    generated = api.call('POST', '/v1/chat/completions', prompt)
    if not generated['choices'][0]['message'].get('content'):
        raise RuntimeError('generation returned empty content')
    for strategy in ('direct', 'agent'):
        result = api.call('POST', '/v1/rag', {'index_id': 'smoke-v1', 'query': {'text': 'vegan spinach dinner'},
                         'strategy': strategy, 'max_tokens': 128})
        if not result['results'] or not result['answer']:
            raise RuntimeError(f'{strategy} RAG returned empty output')
        evidence['checks'][strategy] = {'timings': result['timings'], 'searches': result['searches']}
        if strategy == 'agent' and not result['searches']:
            raise RuntimeError('agent fell back to original query: validate vLLM tool parser before enabling agent mode')
    image_rag = api.call('POST', '/v1/rag', {'index_id': 'smoke-v1', 'query': {'image': image}, 'answer': False})
    if not image_rag['results'] or not image_rag['query_text_used']:
        raise RuntimeError('image-only RAG rewriting/retrieval failed')
    body = {'kind': 'rag', 'payload': {'index_id': 'smoke-v1', 'query': {'text': 'banana'}, 'answer': False}}
    job = api.call('POST', '/v1/jobs', body, key='smoke-rag-' + uuid.uuid4().hex)
    api.wait(job['id'])
    events = api.call('GET', f"/v1/jobs/{job['id']}/events")['events']
    if [e['seq'] for e in events] != list(range(1, len(events) + 1)) or events[-1]['type'] != 'job.succeeded':
        raise RuntimeError('durable event sequence failed')
    req = urllib.request.Request(api.base + f"/v1/jobs/{job['id']}/events/stream",
                                  headers={**api.headers, 'Last-Event-ID': '1'})
    with urllib.request.urlopen(req, timeout=30) as response:
        stream = response.read().decode()
    if 'id: 1\n' in stream or 'job.succeeded' not in stream:
        raise RuntimeError('SSE replay failed')
    evidence['checks']['jobs_and_events'] = {'seed_job_id': seed_id, 'rag_job_id': job['id'], 'sse_replay': True}
    evidence['scope'] = 'Protocol and live model smoke checks; not a model-quality benchmark or load test.'
    save('smoke.json', evidence)


def seed_recipes(api, args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    name = 'mrdbourke/recipe-synthetic-images-10k'
    revision = HfApi().dataset_info(name, revision=args.revision).sha
    dataset = load_dataset(name, revision=revision, split='train')
    n = min(args.limit, len(dataset)) if args.limit else len(dataset)
    spec = {'index_id': args.index, 'expected_documents': n, 'transform': 'recipe_v1',
            'source': f'{name}@{revision}:train:first-{n}:content-cap-32000'}
    api.call('POST', '/v1/indexes', spec)
    print(f'Pinned dataset revision: {revision}. Resume with --revision {revision}.', flush=True)
    truncated, session = 0, uuid.uuid4().hex
    for start in range(0, n, 4):
        docs = []
        for i in range(start, min(n, start + 4)):
            row = dataset[i]
            text = row['recipe_markdown']
            is_truncated = len(text) > 32000
            truncated += int(is_truncated)
            docs.append({'id': str(row['id']), 'title': str(row['name']), 'content': text[:32000],
                         'image': image_url(row['image']),
                         'metadata': {'ordinal': i, 'minutes': str(row['minutes']), 'truncated': is_truncated}})
        batch = {'documents': docs}
        if args.jobs:
            fingerprint = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
            job = api.call('POST', '/v1/jobs', {'kind': 'ingest', 'payload': {'index_id': args.index, 'batch': batch}},
                           key=f'{session}:{start}:{fingerprint}', retries=2)
            api.wait(job['id'])
        else:
            api.call('PUT', f'/v1/indexes/{args.index}/documents', batch, retries=2)
        print(f'{min(n, start + 4)}/{n} indexed', flush=True)
    manifest = api.call('POST', f'/v1/indexes/{args.index}/publish', retries=2)
    save(f'{args.index}-manifest.json', {**manifest, 'truncated_documents': truncated})


def benchmark(api, args):
    body = {'index_id': args.index, 'query': {'text': 'quick vegetarian spinach dinner'},
            'strategy': args.strategy, 'answer': args.answer, 'max_tokens': 128}
    api.call('POST', '/v1/rag', body)  # Excluded warmup.
    def one(_):
        t = time.monotonic()
        try:
            api.call('POST', '/v1/rag', body)
            return {'seconds': time.monotonic() - t, 'status': 200}
        except APIError as exc:
            return {'seconds': time.monotonic() - t, 'status': exc.code}
    t = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, range(args.n)))
    elapsed = time.monotonic() - t
    successes = sorted(r['seconds'] for r in rows if r['status'] == 200)
    result = {'kind': 'closed-loop HTTP benchmark', 'time': time.time(), 'request': body,
              'concurrency': args.concurrency, 'n': args.n, 'failures': args.n - len(successes),
              'successes_per_second': len(successes) / elapsed, 'rows': rows}
    for label, q in [('p50_s', .5), ('p95_s', .95)]:
        result[label] = successes[min(len(successes)-1, math.ceil(q*len(successes))-1)] if successes else None
    save('benchmark.json', result)


def evaluate(api, args):
    query_file = Path(args.queries)
    rows = json.loads(query_file.read_text())[:args.n]
    if not rows:
        raise ValueError('query file is empty')
    manifest = api.call('GET', f'/v1/indexes/{args.index}')
    if manifest['documents'] < 10096:
        raise RuntimeError('Known-item evaluation requires the full 10,096-recipe corpus, not a smoke/partial index')
    catalog, offset = {}, 0
    while offset is not None:
        page = api.call('GET', f'/v1/indexes/{args.index}/documents?offset={offset}&limit=1000')
        catalog.update({d['metadata'].get('ordinal'): d['title'] for d in page['documents']})
        offset = page['next_offset']
    normalized = lambda text: ' '.join(text.casefold().split())
    for row in rows:
        if row['doc_id'] not in catalog or normalized(catalog[row['doc_id']]) != normalized(row['title']):
            raise RuntimeError(f"Evaluation corpus/order mismatch for doc_id={row['doc_id']}; do not score this run")
    output = {'time': time.time(), 'query_sha256': hashlib.sha256(query_file.read_bytes()).hexdigest(),
              'index': manifest, 'answer_generation': False, 'results': {}}
    for query_set in ('title', 'paraphrase', 'es', 'bn', 'zh'):
        for strategy in args.strategies.split(','):
            if strategy not in ('direct', 'agent'):
                raise ValueError('strategies must be direct,agent or one of those')
            observations = []
            for row in rows:
                started = time.monotonic()
                response = api.call('POST', '/v1/rag', {'index_id': args.index, 'query': {'text': row[query_set]},
                                    'strategy': strategy, 'top_k': 10, 'answer': False})
                ranks = [d['metadata'].get('ordinal') for d in response['results']]
                rank = ranks.index(row['doc_id']) + 1 if row['doc_id'] in ranks else None
                observations.append({'doc_id': row['doc_id'], 'rank': rank, 'seconds': time.monotonic() - started})
            n = len(observations)
            stats = {'n': n, 'recall@1': sum(o['rank'] == 1 for o in observations) / n,
                     'recall@10': sum(o['rank'] is not None for o in observations) / n,
                     'mrr@10': sum(1 / o['rank'] for o in observations if o['rank']) / n,
                     'mean_latency_s': sum(o['seconds'] for o in observations) / n, 'queries': observations}
            output['results'][query_set + ' | ' + strategy] = stats
            save('evaluation.json', output)
            print(query_set, strategy, {k: v for k, v in stats.items() if k != 'queries'}, flush=True)


def record_images():
    envfile()
    images = set(command(['docker', 'compose', 'config', '--images']).splitlines())
    result = {}
    for image in sorted(images):
        result[image] = json.loads(command(['docker', 'image', 'inspect', image, '--format',
                                            '{"id":{{json .Id}},"digests":{{json .RepoDigests}}}']))
    save('images.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init-env')
    p = sub.add_parser('doctor'); p.add_argument('--containers', action='store_true')
    sub.add_parser('smoke'); sub.add_parser('record-images')
    p = sub.add_parser('seed-recipes')
    p.add_argument('--index', default='recipes-v1'); p.add_argument('--revision', default='main')
    p.add_argument('--limit', type=int, default=0); p.add_argument('--jobs', action='store_true')
    p = sub.add_parser('benchmark')
    p.add_argument('--index', default='recipes-v1'); p.add_argument('--n', type=int, default=20)
    p.add_argument('--concurrency', type=int, default=1); p.add_argument('--strategy', choices=['direct','agent'], default='direct')
    p.add_argument('--answer', action='store_true')
    p = sub.add_parser('eval')
    p.add_argument('--index', default='recipes-v1'); p.add_argument('--n', type=int, default=50)
    p.add_argument('--queries', default=str(HERE.parent / 'results' / 'eval_queries.json'))
    p.add_argument('--strategies', default='direct,agent')
    args = parser.parse_args()
    if hasattr(args, 'n') and args.n < 1: parser.error('--n must be positive')
    if hasattr(args, 'limit') and args.limit < 0: parser.error('--limit must be >=0')
    if hasattr(args, 'concurrency') and not 1 <= args.concurrency <= 32: parser.error('--concurrency must be 1..32')
    if args.command == 'init-env': return init_env()
    if args.command == 'doctor': return doctor(args.containers)
    if args.command == 'record-images': return record_images()
    api = API(args.base_url)
    if args.command == 'smoke': return smoke(api)
    if args.command == 'seed-recipes': return seed_recipes(api, args)
    if args.command == 'benchmark': return benchmark(api, args)
    if args.command == 'eval': return evaluate(api, args)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
