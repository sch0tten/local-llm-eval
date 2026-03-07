#!/usr/bin/env python3
"""
Swarm Benchmark — Concurrency Scaling (1 to 4 simultaneous agents)

Architecture (client-server, declared):
  ┌─────────────────────┐         LAN         ┌──────────────────────┐
  │  Workstation         │ ──── 1 Gbps ─────> │  Inference Server    │
  │  (this script)       │                     │  vLLM on :8000       │
  │  orchestrates here   │                     │  2x RTX 3090 NVLink  │
  │                      │                     │  48GB VRAM           │
  └─────────────────────┘                     └──────────────────────┘

  This benchmark runs on the workstation and fires API requests over the
  internal network to the inference server — exactly as Claude Code subagents
  would in production. The network hop is part of the measurement.

Prompt Design:
  16 unique prompts: 4 task types × 4 variants (one variant per concurrency
  level). This eliminates vLLM prefix caching as a confounding variable —
  every request at every concurrency level hits cold prefill.

  Prompts are scoped to single-module output (~2000-4000 tokens). This is a
  deliberate methodology choice: coding agent subagents produce focused,
  single-file outputs — one module, one test file, one refactored class.
  Requesting multi-file packages would exceed max_tokens and introduce
  truncation as a confounding variable. By targeting natural completion
  (finish_reason: stop) for all prompts, we ensure tok/s measurements
  reflect sustained generation, not early cutoff.

Usage:
  python3 bench-swarm.py --base-url http://INFERENCE_SERVER:8000/v1 --model MODEL

Flags:
  --max-tokens     Max completion tokens (default: 8192)
  --temperature    Sampling temperature (default: 0.7)
  --runs           Measurement runs per concurrency level (default: 2)
  --warmup         Warmup runs before measurement (default: 1)
  --concurrency    Concurrency levels to test (default: 1 2 3 4)
  --output-csv     Path to write CSV results
"""

import argparse
import asyncio
import csv
import time
from dataclasses import dataclass, field

# =============================================================================
# 16 unique coding prompts: 4 task types × 4 variants
#
# Each concurrency level uses a DIFFERENT variant set to defeat prefix caching.
# All variants within a task type target equivalent complexity and output size.
# Prompts target single-module output to ensure finish_reason: stop at 8192.
# =============================================================================

TASK_TYPES = ["algorithm", "testing", "refactoring", "system_design"]

PROMPT_SETS = {
    # ── C=1: Variant A ──────────────────────────────────────────────────
    1: [
        {
            "key": "algo_a",
            "label": "LRU Cache",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement an LRU cache in Python as a single module.\n"
                "Requirements:\n"
                "1. O(1) get(key) and put(key, value) using a doubly-linked list + dict\n"
                "2. Thread-safe with threading.Lock\n"
                "3. Optional TTL per entry with lazy expiration on access\n"
                "4. Eviction callback hook called when entries are removed\n"
                "5. Type hints and docstrings\n"
                "6. Statistics: hits, misses, evictions, hit_rate property\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_a",
            "label": "Pytest: Cron Parser",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a cron expression parser with this interface:\n"
                "```python\n"
                "def parse_cron(expr: str) -> dict[str, list[int]]\n"
                "```\n"
                "It parses standard 5-field cron (minute, hour, day-of-month, month, day-of-week) "
                "and returns expanded integer lists for each field.\n"
                "Cover with parametrize: wildcards (*), ranges (1-5), steps (*/15), "
                "comma lists (1,3,5), combined expressions, edge cases (boundaries, "
                "invalid fields, empty input, too many/few fields), and at least 15 test cases."
            )}],
        },
        {
            "key": "refactor_a",
            "label": "Refactor: Order Endpoint",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this Flask route into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.route('/api/orders', methods=['POST'])\n"
                "def create_order():\n"
                "    data = request.get_json()\n"
                "    user = db.session.query(User).filter_by(id=data['user_id']).first()\n"
                "    if not user:\n"
                "        return jsonify({'error': 'User not found'}), 404\n"
                "    if user.balance < data['total']:\n"
                "        return jsonify({'error': 'Insufficient balance'}), 400\n"
                "    order = Order(user_id=user.id, total=data['total'], items=data['items'])\n"
                "    user.balance -= data['total']\n"
                "    db.session.add(order)\n"
                "    db.session.commit()\n"
                "    send_email(user.email, 'Order confirmed', f'Order {order.id}')\n"
                "    return jsonify({'order_id': order.id}), 201\n"
                "```\n\n"
                "Produce in one file: repository classes, service layer with transaction "
                "boundaries, Pydantic request/response models, custom exceptions with "
                "HTTP status mapping, and the simplified route handler. Include type hints."
            )}],
        },
        {
            "key": "design_a",
            "label": "Design: Rate Limiter",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a rate limiter in a single Python module:\n"
                "1. Token bucket algorithm with configurable rate and burst\n"
                "2. Per-key limiting (e.g., per API client)\n"
                "3. Thread-safe operation\n"
                "4. Decorator for Flask/FastAPI routes\n"
                "5. Returns rate limit headers (X-RateLimit-Limit, Remaining, Reset)\n"
                "6. Type hints and docstrings\n"
                "7. Usage example at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=2: Variant B ──────────────────────────────────────────────────
    2: [
        {
            "key": "algo_b",
            "label": "Priority Task Queue",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a thread-safe priority task queue in Python as a single module.\n"
                "Requirements:\n"
                "1. O(log n) insert, O(log n) extract-min using heapq\n"
                "2. Tasks have priority (int), deadline (datetime), callable, and unique task_id\n"
                "3. Thread-safe with threading.Lock\n"
                "4. cancel_task(task_id) with O(1) lookup via auxiliary dict\n"
                "5. Deadline-aware: expired tasks are skipped on extraction\n"
                "6. Type hints and docstrings\n"
                "7. Statistics: tasks_completed, tasks_expired, tasks_cancelled\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_b",
            "label": "Pytest: JWT Validator",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a JWT token validator with this interface:\n"
                "```python\n"
                "class JWTValidator:\n"
                "    def __init__(self, secret: str, algorithms: list[str] = ['HS256']): ...\n"
                "    def validate(self, token: str) -> dict: ...\n"
                "    def is_expired(self, token: str) -> bool: ...\n"
                "```\n"
                "Cover with parametrize: valid tokens (HS256), expired tokens (exp claim), "
                "not-before (nbf), issuer validation (iss), audience validation (aud), "
                "malformed tokens (missing segments, bad base64, corrupted signature), "
                "empty payload, clock skew tolerance, custom claims, invalid algorithm. "
                "At least 15 test cases. Include fixtures for token generation."
            )}],
        },
        {
            "key": "refactor_b",
            "label": "Refactor: Registration View",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this Django view into service + repository layers in a single file:\n\n"
                "```python\n"
                "def register_user(request):\n"
                "    if request.method != 'POST':\n"
                "        return JsonResponse({'error': 'Method not allowed'}, status=405)\n"
                "    data = json.loads(request.body)\n"
                "    if User.objects.filter(email=data['email']).exists():\n"
                "        return JsonResponse({'error': 'Email already registered'}, status=409)\n"
                "    if len(data['password']) < 8:\n"
                "        return JsonResponse({'error': 'Password too short'}, status=400)\n"
                "    user = User.objects.create_user(\n"
                "        username=data['email'], email=data['email'],\n"
                "        password=data['password'], first_name=data.get('name', '')\n"
                "    )\n"
                "    profile = UserProfile.objects.create(user=user, role='member')\n"
                "    token = EmailVerificationToken.objects.create(user=user)\n"
                "    send_mail('Verify', f'/verify/{token.token}', 'no-reply@app.com', [user.email])\n"
                "    return JsonResponse({'user_id': user.id}, status=201)\n"
                "```\n\n"
                "Produce in one file: repository classes, service layer with transaction, "
                "Pydantic models for request/response, custom exceptions (DuplicateEmail, "
                "WeakPassword) with HTTP mapping, and the simplified view. Include type hints."
            )}],
        },
        {
            "key": "design_b",
            "label": "Design: Circuit Breaker",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a circuit breaker in a single Python module:\n"
                "1. Three states: closed, open, half-open with configurable thresholds\n"
                "2. Failure counting with sliding window (time-based, not just total)\n"
                "3. Decorator API: @circuit_breaker(failure_threshold=5, recovery_timeout=30)\n"
                "4. Fallback function support when circuit is open\n"
                "5. Event hooks: on_open, on_close, on_half_open\n"
                "6. Thread-safe state transitions\n"
                "7. Type hints and docstrings\n"
                "8. Usage example at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=3: Variant C ──────────────────────────────────────────────────
    3: [
        {
            "key": "algo_c",
            "label": "Skip List",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a skip list in Python as a single module.\n"
                "Requirements:\n"
                "1. O(log n) average insert, delete, and search\n"
                "2. Probabilistic balancing with configurable probability (default p=0.5)\n"
                "3. Range query: range(start, end) returns elements in [start, end)\n"
                "4. Rank operation: get_by_rank(k) returns k-th smallest element\n"
                "5. Iterator support (__iter__) for ordered traversal\n"
                "6. Type hints and docstrings\n"
                "7. Statistics: level_count, element_count, avg_search_depth\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_c",
            "label": "Pytest: URL Router",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a URL router with this interface:\n"
                "```python\n"
                "class Router:\n"
                "    def add_route(self, method: str, pattern: str, handler: Callable) -> None: ...\n"
                "    def match(self, method: str, path: str) -> tuple[Callable, dict] | None: ...\n"
                "```\n"
                "Cover with parametrize: static paths (/users), parameterized paths "
                "(/users/{id}), typed params (/users/{id:int}), wildcard (/files/{path:path}), "
                "HTTP method matching (GET vs POST), 405 vs 404 distinction, "
                "trailing slash normalization, conflicting routes, route priority, "
                "unicode paths, duplicate registration, case sensitivity. "
                "At least 15 test cases. Include fixtures and clear docstrings."
            )}],
        },
        {
            "key": "refactor_c",
            "label": "Refactor: Payment Endpoint",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this FastAPI endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.post('/api/payments')\n"
                "async def process_payment(request: Request):\n"
                "    data = await request.json()\n"
                "    order = await db.fetch_one('SELECT * FROM orders WHERE id = :id', {'id': data['order_id']})\n"
                "    if not order:\n"
                "        raise HTTPException(404, 'Order not found')\n"
                "    if order['status'] != 'pending':\n"
                "        raise HTTPException(400, 'Order already processed')\n"
                "    result = stripe.PaymentIntent.create(\n"
                "        amount=int(order['total'] * 100), currency='usd',\n"
                "        payment_method=data['payment_method_id'], confirm=True\n"
                "    )\n"
                "    if result.status == 'succeeded':\n"
                "        await db.execute('UPDATE orders SET status=:s WHERE id=:id',\n"
                "                        {'s': 'paid', 'id': order['id']})\n"
                "        return {'status': 'success', 'payment_id': result.id}\n"
                "    raise HTTPException(402, 'Payment failed')\n"
                "```\n\n"
                "Produce in one file: repository class, service layer with Stripe behind "
                "a gateway interface, Pydantic models, custom exceptions (OrderNotFound, "
                "PaymentFailed) with HTTP mapping, and simplified route. Include type hints."
            )}],
        },
        {
            "key": "design_c",
            "label": "Design: Job Queue",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement an in-process job queue in a single Python module:\n"
                "1. Priority-based scheduling with configurable worker count\n"
                "2. Job lifecycle: pending → running → completed/failed/retrying\n"
                "3. Retry with configurable backoff (linear or exponential)\n"
                "4. Dead letter list for jobs exceeding max_retries\n"
                "5. Both sync and async job handlers\n"
                "6. Graceful shutdown: finish running jobs on SIGTERM\n"
                "7. Thread-safe with proper worker pool management\n"
                "8. Type hints and docstrings\n"
                "9. Usage example at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=4: Variant D ──────────────────────────────────────────────────
    4: [
        {
            "key": "algo_d",
            "label": "Concurrent Hash Map",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a concurrent hash map in Python as a single module.\n"
                "Requirements:\n"
                "1. Segment-based locking (16 segments) for fine-grained concurrency\n"
                "2. O(1) average get, put, delete with automatic resize at load factor > 0.75\n"
                "3. Atomic compute_if_absent(key, factory) method\n"
                "4. Iterator provides a weakly-consistent snapshot view\n"
                "5. Type hints and docstrings\n"
                "6. Statistics: size, collision_count, resize_count per segment\n"
                "7. Bulk put_all(mapping) that minimizes lock acquisitions\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_d",
            "label": "Pytest: Config Parser",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a hierarchical config parser with this interface:\n"
                "```python\n"
                "class Config:\n"
                "    def __init__(self, source: str | Path | dict): ...\n"
                "    def get(self, dotted_key: str, default: Any = MISSING) -> Any: ...\n"
                "    def merge(self, other: 'Config') -> 'Config': ...\n"
                "    def to_dict(self) -> dict: ...\n"
                "```\n"
                "Cover with parametrize: dotted key access (db.host), nested merge behavior, "
                "environment variable interpolation ($ENV{VAR}), default values, "
                "type coercion, missing keys with and without default, "
                "malformed input, unicode keys/values, empty configs, "
                "deep merge vs shallow merge, to_dict round-trip. "
                "At least 15 test cases. Include fixtures with tmp_path."
            )}],
        },
        {
            "key": "refactor_d",
            "label": "Refactor: Inventory Transfer",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this Flask endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.route('/api/inventory/transfer', methods=['POST'])\n"
                "def transfer_inventory():\n"
                "    data = request.get_json()\n"
                "    src = db.session.query(Warehouse).get(data['source_id'])\n"
                "    dst = db.session.query(Warehouse).get(data['dest_id'])\n"
                "    if not src or not dst:\n"
                "        return jsonify({'error': 'Warehouse not found'}), 404\n"
                "    for item in data['items']:\n"
                "        inv = db.session.query(Inventory).filter_by(\n"
                "            warehouse_id=src.id, product_id=item['product_id']).first()\n"
                "        if not inv or inv.quantity < item['quantity']:\n"
                "            db.session.rollback()\n"
                "            return jsonify({'error': 'Insufficient stock'}), 400\n"
                "        inv.quantity -= item['quantity']\n"
                "    db.session.commit()\n"
                "    return jsonify({'status': 'transferred'}), 200\n"
                "```\n\n"
                "Produce in one file: repository classes, service layer with explicit "
                "transaction + rollback, Pydantic models, custom exceptions "
                "(WarehouseNotFound, InsufficientStock) with HTTP mapping, "
                "and simplified route. Include type hints."
            )}],
        },
        {
            "key": "design_d",
            "label": "Design: Service Registry",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a service registry and health checker in a single Python module:\n"
                "1. Register services with metadata (name, host, port, tags)\n"
                "2. Health checks: HTTP GET and TCP connect with configurable interval\n"
                "3. Service states: healthy, unhealthy, deregistered\n"
                "4. Load-balanced lookup: get_service(name) returns healthy instance (round-robin)\n"
                "5. Watch API: subscribe to state changes with callbacks\n"
                "6. Auto-deregister after TTL without heartbeat\n"
                "7. Thread-safe registry\n"
                "8. Type hints and docstrings\n"
                "9. Usage example at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=5: Variant E ──────────────────────────────────────────────────
    5: [
        {
            "key": "algo_e",
            "label": "B-Tree",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a B-tree in Python as a single module.\n"
                "Requirements:\n"
                "1. Configurable order (minimum degree t, default t=3)\n"
                "2. O(log n) search, insert, and delete with node splitting and merging\n"
                "3. In-order traversal iterator (__iter__)\n"
                "4. Bulk load from sorted sequence for optimal initial tree\n"
                "5. Pretty-print showing tree structure level by level\n"
                "6. Type hints and docstrings\n"
                "7. Statistics: height, node_count, fill_factor (avg keys per node / max)\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_e",
            "label": "Pytest: Markdown Parser",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a Markdown-to-HTML converter with this interface:\n"
                "```python\n"
                "def md_to_html(text: str) -> str\n"
                "```\n"
                "Cover with parametrize: headings (h1-h6), bold (**), italic (*), "
                "inline code (`), code blocks (```), links [text](url), images, "
                "unordered lists, ordered lists, blockquotes, horizontal rules, "
                "nested formatting (bold inside italic), escaped characters, "
                "empty input, multiline paragraphs. At least 18 test cases."
            )}],
        },
        {
            "key": "refactor_e",
            "label": "Refactor: File Upload",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this FastAPI endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.post('/api/files/upload')\n"
                "async def upload_file(file: UploadFile, user_id: int = Form(...)):\n"
                "    if file.size > 10_000_000:\n"
                "        raise HTTPException(413, 'File too large')\n"
                "    ext = file.filename.rsplit('.', 1)[-1].lower()\n"
                "    if ext not in ('pdf', 'png', 'jpg', 'docx'):\n"
                "        raise HTTPException(400, 'Unsupported file type')\n"
                "    content = await file.read()\n"
                "    hash_val = hashlib.sha256(content).hexdigest()\n"
                "    existing = db.query(FileRecord).filter_by(hash=hash_val).first()\n"
                "    if existing:\n"
                "        return {'file_id': existing.id, 'status': 'duplicate'}\n"
                "    path = f'/storage/{user_id}/{hash_val}.{ext}'\n"
                "    async with aiofiles.open(path, 'wb') as f:\n"
                "        await f.write(content)\n"
                "    record = FileRecord(user_id=user_id, filename=file.filename,\n"
                "                        hash=hash_val, path=path, size=file.size)\n"
                "    db.add(record)\n"
                "    db.commit()\n"
                "    return {'file_id': record.id, 'status': 'uploaded'}\n"
                "```\n\n"
                "Produce in one file: storage backend interface, repository class, service layer "
                "with validation, Pydantic models, custom exceptions (FileTooLarge, "
                "UnsupportedType, DuplicateFile) with HTTP mapping, and simplified route. "
                "Include type hints."
            )}],
        },
        {
            "key": "design_e",
            "label": "Design: Event Sourcing",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement an event sourcing system in a single Python module:\n"
                "1. Event store with append-only log (in-memory, serializable to JSON)\n"
                "2. Aggregate base class that rebuilds state from events\n"
                "3. Snapshot support: save/restore aggregate state at a given version\n"
                "4. Event bus with sync subscribers (projections)\n"
                "5. Optimistic concurrency control (expected version on append)\n"
                "6. Example: BankAccount aggregate with Deposited, Withdrawn, Frozen events\n"
                "7. Thread-safe event store\n"
                "8. Type hints and docstrings\n"
                "9. Usage example at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=6: Variant F ──────────────────────────────────────────────────
    6: [
        {
            "key": "algo_f",
            "label": "Trie Autocomplete",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a trie with autocomplete in Python as a single module.\n"
                "Requirements:\n"
                "1. O(k) insert and search where k is key length\n"
                "2. Prefix search: all_with_prefix(prefix) returns matching keys\n"
                "3. Top-k autocomplete by frequency: suggest(prefix, k=5) returns most frequent matches\n"
                "4. Wildcard search: search_pattern('c?t') matches 'cat', 'cut'\n"
                "5. Delete with trie compaction (remove empty branches)\n"
                "6. Serialization to/from dict for persistence\n"
                "7. Type hints and docstrings\n"
                "8. Statistics: word_count, node_count, avg_depth\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_f",
            "label": "Pytest: SQL Builder",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a SQL query builder with this interface:\n"
                "```python\n"
                "class Query:\n"
                "    def select(self, *columns: str) -> 'Query': ...\n"
                "    def from_(self, table: str) -> 'Query': ...\n"
                "    def where(self, condition: str, **params) -> 'Query': ...\n"
                "    def join(self, table: str, on: str) -> 'Query': ...\n"
                "    def order_by(self, column: str, desc: bool = False) -> 'Query': ...\n"
                "    def limit(self, n: int) -> 'Query': ...\n"
                "    def build(self) -> tuple[str, dict]: ...\n"
                "```\n"
                "Cover with parametrize: simple select, multiple columns, WHERE with params, "
                "multiple WHERE (AND), JOIN, LEFT JOIN, ORDER BY asc/desc, LIMIT, "
                "chained operations, subquery in WHERE, GROUP BY + HAVING, "
                "SQL injection prevention (params), empty query, missing FROM, "
                "complex multi-join. At least 18 test cases."
            )}],
        },
        {
            "key": "refactor_f",
            "label": "Refactor: Notification Endpoint",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this Flask endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.route('/api/notifications/send', methods=['POST'])\n"
                "def send_notification():\n"
                "    data = request.get_json()\n"
                "    users = db.session.query(User).filter(\n"
                "        User.id.in_(data['user_ids'])).all()\n"
                "    if len(users) != len(data['user_ids']):\n"
                "        missing = set(data['user_ids']) - {u.id for u in users}\n"
                "        return jsonify({'error': f'Users not found: {missing}'}), 404\n"
                "    results = []\n"
                "    for user in users:\n"
                "        pref = db.session.query(NotificationPref).filter_by(\n"
                "            user_id=user.id).first()\n"
                "        if pref and pref.channel == 'email':\n"
                "            send_email(user.email, data['title'], data['body'])\n"
                "        elif pref and pref.channel == 'sms':\n"
                "            send_sms(user.phone, data['body'])\n"
                "        else:\n"
                "            send_push(user.device_token, data['title'], data['body'])\n"
                "        log = NotificationLog(user_id=user.id, channel=pref.channel if pref else 'push',\n"
                "                              title=data['title'], status='sent')\n"
                "        db.session.add(log)\n"
                "        results.append({'user_id': user.id, 'channel': log.channel})\n"
                "    db.session.commit()\n"
                "    return jsonify({'sent': results}), 200\n"
                "```\n\n"
                "Produce in one file: channel strategy pattern (email/sms/push), repository, "
                "service with batch processing, Pydantic models, custom exceptions "
                "(UsersNotFound, DeliveryFailed) with HTTP mapping, and simplified route. "
                "Include type hints."
            )}],
        },
        {
            "key": "design_f",
            "label": "Design: Plugin System",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a plugin system in a single Python module:\n"
                "1. Plugin discovery: scan a directory for Python files implementing a base class\n"
                "2. Plugin lifecycle: load, initialize, enable, disable, unload\n"
                "3. Dependency resolution: plugins declare dependencies, loaded in topological order\n"
                "4. Hook system: plugins register hooks, host app calls them in priority order\n"
                "5. Configuration: each plugin has a typed config schema (dataclass)\n"
                "6. Sandboxed execution: catch plugin exceptions without crashing host\n"
                "7. Thread-safe plugin registry\n"
                "8. Type hints and docstrings\n"
                "9. Example: two plugins (logging + metrics) with a dependency\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=7: Variant G ──────────────────────────────────────────────────
    7: [
        {
            "key": "algo_g",
            "label": "Disjoint Set Union",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a Disjoint Set Union (Union-Find) in Python as a single module.\n"
                "Requirements:\n"
                "1. O(α(n)) amortized union and find with path compression + union by rank\n"
                "2. Weighted variant: each set tracks a cumulative weight\n"
                "3. Rollback support: undo the last k unions (persistent stack)\n"
                "4. Connected components: iterate all components with their members\n"
                "5. Merge callback: optional hook called when two sets merge\n"
                "6. Type hints and docstrings\n"
                "7. Statistics: component_count, largest_component_size, total_unions\n"
                "8. Example: Kruskal's MST at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_g",
            "label": "Pytest: CSV Processor",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a CSV processor with this interface:\n"
                "```python\n"
                "class CSVProcessor:\n"
                "    def __init__(self, path: Path): ...\n"
                "    def read(self, encoding: str = 'utf-8') -> list[dict]: ...\n"
                "    def filter(self, predicate: Callable[[dict], bool]) -> list[dict]: ...\n"
                "    def transform(self, column: str, func: Callable) -> list[dict]: ...\n"
                "    def aggregate(self, column: str, func: str) -> float: ...\n"
                "    def write(self, rows: list[dict], path: Path) -> None: ...\n"
                "```\n"
                "Cover with parametrize: basic read, unicode content, empty file, "
                "missing columns, filter by value, filter by multiple conditions, "
                "transform column types, aggregate sum/avg/min/max, write round-trip, "
                "quoted fields with commas, newlines in quoted fields, "
                "BOM handling, large row count, header-only file, mismatched columns. "
                "At least 18 test cases. Use tmp_path fixture."
            )}],
        },
        {
            "key": "refactor_g",
            "label": "Refactor: Search Endpoint",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this FastAPI endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.get('/api/search')\n"
                "async def search_products(q: str, category: str = None,\n"
                "                          min_price: float = None, max_price: float = None,\n"
                "                          page: int = 1, size: int = 20):\n"
                "    query = db.query(Product).filter(Product.name.ilike(f'%{q}%'))\n"
                "    if category:\n"
                "        query = query.filter(Product.category == category)\n"
                "    if min_price is not None:\n"
                "        query = query.filter(Product.price >= min_price)\n"
                "    if max_price is not None:\n"
                "        query = query.filter(Product.price <= max_price)\n"
                "    total = query.count()\n"
                "    items = query.offset((page - 1) * size).limit(size).all()\n"
                "    return {\n"
                "        'items': [{'id': p.id, 'name': p.name, 'price': float(p.price),\n"
                "                   'category': p.category} for p in items],\n"
                "        'total': total, 'page': page, 'pages': (total + size - 1) // size\n"
                "    }\n"
                "```\n\n"
                "Produce in one file: filter specification pattern, repository with dynamic "
                "query building, service layer, Pydantic models (SearchRequest, SearchResponse "
                "with pagination), and simplified route. Include type hints."
            )}],
        },
        {
            "key": "design_g",
            "label": "Design: State Machine",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a finite state machine framework in a single Python module:\n"
                "1. Declarative state/transition definition via decorators\n"
                "2. Guard conditions: transitions only fire if guard returns True\n"
                "3. Entry/exit actions per state\n"
                "4. Hierarchical states (substates with parent fallback)\n"
                "5. Event queue with async processing\n"
                "6. Transition history log with timestamps\n"
                "7. Visualization: export to DOT format for Graphviz\n"
                "8. Thread-safe state transitions\n"
                "9. Type hints and docstrings\n"
                "10. Example: order lifecycle (pending→paid→shipped→delivered/returned)\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],

    # ── C=8: Variant H ──────────────────────────────────────────────────
    8: [
        {
            "key": "algo_h",
            "label": "Bloom Filter",
            "task_type": "algorithm",
            "messages": [{"role": "user", "content": (
                "Implement a Bloom filter in Python as a single module.\n"
                "Requirements:\n"
                "1. Configurable expected elements and false positive rate\n"
                "2. Automatic optimal bit array size and hash count calculation\n"
                "3. Counting variant: support delete operations with 4-bit counters\n"
                "4. Scalable: auto-grow by adding new filter layers when capacity reached\n"
                "5. Union and intersection of two Bloom filters\n"
                "6. Serialization to/from bytes for persistence\n"
                "7. Type hints and docstrings\n"
                "8. Statistics: estimated_count, fill_ratio, false_positive_estimate\n"
                "Provide the complete single-file implementation."
            )}],
        },
        {
            "key": "test_h",
            "label": "Pytest: Cache Decorator",
            "task_type": "testing",
            "messages": [{"role": "user", "content": (
                "Write a pytest test suite for a caching decorator with this interface:\n"
                "```python\n"
                "def cached(maxsize: int = 128, ttl: float = None,\n"
                "           key_func: Callable = None) -> Callable:\n"
                "    '''Decorator that caches function results.'''\n"
                "    ...\n"
                "# Decorated functions get these extra attributes:\n"
                "# f.cache_info() -> CacheInfo(hits, misses, size, maxsize)\n"
                "# f.cache_clear() -> None\n"
                "```\n"
                "Cover with parametrize: basic caching, cache hit/miss counting, "
                "maxsize eviction (LRU), TTL expiration, custom key function, "
                "unhashable arguments, concurrent access, cache_clear, cache_info, "
                "zero maxsize (no cache), None ttl (no expiry), decorated method on class, "
                "recursive function, exception not cached, kwargs ordering. "
                "At least 18 test cases. Use time.sleep or freezegun for TTL tests."
            )}],
        },
        {
            "key": "refactor_h",
            "label": "Refactor: Booking Endpoint",
            "task_type": "refactoring",
            "messages": [{"role": "user", "content": (
                "Refactor this Flask endpoint into service + repository layers in a single file:\n\n"
                "```python\n"
                "@app.route('/api/bookings', methods=['POST'])\n"
                "def create_booking():\n"
                "    data = request.get_json()\n"
                "    room = db.session.query(Room).get(data['room_id'])\n"
                "    if not room:\n"
                "        return jsonify({'error': 'Room not found'}), 404\n"
                "    start = datetime.fromisoformat(data['start'])\n"
                "    end = datetime.fromisoformat(data['end'])\n"
                "    if end <= start:\n"
                "        return jsonify({'error': 'Invalid time range'}), 400\n"
                "    conflict = db.session.query(Booking).filter(\n"
                "        Booking.room_id == room.id,\n"
                "        Booking.start < end, Booking.end > start,\n"
                "        Booking.status != 'cancelled'\n"
                "    ).first()\n"
                "    if conflict:\n"
                "        return jsonify({'error': 'Room already booked', 'conflict_id': conflict.id}), 409\n"
                "    booking = Booking(room_id=room.id, user_id=data['user_id'],\n"
                "                      start=start, end=end, status='confirmed')\n"
                "    db.session.add(booking)\n"
                "    db.session.commit()\n"
                "    send_confirmation(data['user_id'], booking.id)\n"
                "    return jsonify({'booking_id': booking.id, 'status': 'confirmed'}), 201\n"
                "```\n\n"
                "Produce in one file: repository with conflict detection query, service layer "
                "with booking validation and transaction, Pydantic models, custom exceptions "
                "(RoomNotFound, InvalidTimeRange, RoomConflict) with HTTP mapping, "
                "and simplified route. Include type hints."
            )}],
        },
        {
            "key": "design_h",
            "label": "Design: Task Pipeline",
            "task_type": "system_design",
            "messages": [{"role": "user", "content": (
                "Implement a task pipeline framework in a single Python module:\n"
                "1. DAG-based task dependencies (tasks declare inputs/outputs)\n"
                "2. Parallel execution of independent tasks with thread pool\n"
                "3. Task retry with configurable strategy (immediate, linear, exponential)\n"
                "4. Progress tracking: percentage, ETA, current stage\n"
                "5. Checkpoint/resume: save pipeline state to disk, resume from last checkpoint\n"
                "6. Dry-run mode: validate DAG and show execution plan without running\n"
                "7. Type-safe task inputs/outputs with runtime validation\n"
                "8. Type hints and docstrings\n"
                "9. Example: ETL pipeline (extract→transform→validate→load) at the bottom\n"
                "Provide the complete single-file implementation."
            )}],
        },
    ],
}

# Warmup uses separate short prompts — no overlap with any measurement variant
WARMUP_PROMPTS = [
    {"key": "warmup_1", "label": "Warmup: LinkedList", "task_type": "warmup",
     "messages": [{"role": "user", "content": "Write a Python function that reverses a singly linked list iteratively. Include type hints and a Node class."}]},
    {"key": "warmup_2", "label": "Warmup: Email Tests", "task_type": "warmup",
     "messages": [{"role": "user", "content": "Write 5 pytest tests for a function `def is_valid_email(s: str) -> bool`. Use parametrize."}]},
    {"key": "warmup_3", "label": "Warmup: Refactor", "task_type": "warmup",
     "messages": [{"role": "user", "content": "Refactor this into a class:\n```python\ndef process(data):\n    result = []\n    for item in data:\n        if item['type'] == 'A': result.append(item['value'] * 2)\n        elif item['type'] == 'B': result.append(item['value'] + 10)\n    return result\n```"}]},
    {"key": "warmup_4", "label": "Warmup: PubSub", "task_type": "warmup",
     "messages": [{"role": "user", "content": "Implement a simple thread-safe pub/sub message broker class in Python with subscribe, publish, and unsubscribe methods."}]},
]


@dataclass
class TaskResult:
    prompt_key: str
    label: str
    task_type: str
    completion_tokens: int
    wall_seconds: float
    tok_per_sec: float
    finish_reason: str


@dataclass
class RunResult:
    concurrency: int
    run_number: int
    tasks: list[TaskResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    aggregate_tokens: int = 0
    effective_tok_per_sec: float = 0.0


async def call_completion(
    session, base_url: str, model: str, messages: list, max_tokens: int, temperature: float
) -> tuple[int, float, str]:
    """Single chat completion call over the network to inference server."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    elapsed = time.perf_counter() - t0

    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    return completion_tokens, elapsed, finish_reason


async def run_prompts(
    session, base_url: str, model: str, max_tokens: int, temperature: float,
    prompts: list[dict], concurrency: int
) -> list[TaskResult]:
    """Dispatch prompts at the given concurrency level.

    At concurrency=1, tasks run sequentially (one at a time).
    At concurrency=N, tasks fire in batches of N simultaneously.
    All 4 prompts in the set always run.
    """

    async def _run_one(prompt):
        tokens, elapsed, finish = await call_completion(
            session, base_url, model, prompt["messages"], max_tokens, temperature
        )
        tps = tokens / elapsed if elapsed > 0 else 0
        return TaskResult(
            prompt_key=prompt["key"],
            label=prompt["label"],
            task_type=prompt["task_type"],
            completion_tokens=tokens,
            wall_seconds=elapsed,
            tok_per_sec=tps,
            finish_reason=finish,
        )

    results = []
    for i in range(0, len(prompts), concurrency):
        batch = prompts[i:i + concurrency]
        batch_results = await asyncio.gather(*[_run_one(p) for p in batch])
        results.extend(batch_results)

    for r in results:
        trunc = " TRUNCATED" if r.finish_reason == "length" else ""
        print(f"    [{r.label:<28s}] {r.completion_tokens:5d} tok  {r.wall_seconds:6.1f}s  "
              f"{r.tok_per_sec:6.1f} tok/s  ({r.finish_reason}){trunc}")

    return results


async def main():
    import aiohttp

    parser = argparse.ArgumentParser(
        description="Swarm Benchmark: workstation (orchestrator) → LAN → inference server")
    parser.add_argument("--base-url", required=True,
                        help="vLLM API on inference server (e.g. http://192.168.1.100:8000/v1)")
    parser.add_argument("--model", required=True, help="Model name as served by vLLM")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--runs", type=int, default=2, help="Measurement runs per concurrency level")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (discarded)")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Concurrency levels to test (default: 1 2 3 4 5 6 7 8)")
    parser.add_argument("--output-csv", default="results_swarm.csv")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"SWARM BENCHMARK — Concurrency Scaling")
    print(f"{'='*72}")
    print(f"  Model:        {args.model}")
    print(f"  API endpoint: {args.base_url}")
    print(f"  Architecture: workstation ──LAN──> inference server")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  max_tokens:   {args.max_tokens}")
    print(f"  temperature:  {args.temperature}")
    print(f"  Prompts:      32 unique (4 types × 8 variants, no prefix cache reuse)")
    print(f"  Warmup:       {args.warmup} run(s), measurement: {args.runs} run(s)")
    print(f"{'='*72}")

    all_runs: list[RunResult] = []
    truncations: list[str] = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        # --- warmup: separate prompts at max concurrency ---
        for w in range(args.warmup):
            max_c = max(args.concurrency)
            print(f"\n[Warmup {w+1}/{args.warmup}] concurrency={max_c} (dedicated warmup prompts)")
            await run_prompts(
                session, args.base_url, args.model, args.max_tokens, args.temperature,
                WARMUP_PROMPTS, max_c
            )

        # --- measurement: each concurrency level with its own prompt variant ---
        for c in args.concurrency:
            prompts = PROMPT_SETS[c]
            for r in range(args.runs):
                run_num = r + 1
                print(f"\n[C={c}  Run {run_num}/{args.runs}]  variant {'ABCDEFGH'[c-1]}")

                t0 = time.perf_counter()
                tasks = await run_prompts(
                    session, args.base_url, args.model, args.max_tokens, args.temperature,
                    prompts, c
                )
                wall = time.perf_counter() - t0
                total_tokens = sum(t.completion_tokens for t in tasks)
                eff_tps = total_tokens / wall if wall > 0 else 0

                # Track truncations
                for t in tasks:
                    if t.finish_reason == "length":
                        truncations.append(f"  C={c} Run {run_num}: {t.label} ({t.completion_tokens} tok)")

                rr = RunResult(
                    concurrency=c,
                    run_number=run_num,
                    tasks=tasks,
                    wall_seconds=wall,
                    aggregate_tokens=total_tokens,
                    effective_tok_per_sec=eff_tps,
                )
                all_runs.append(rr)
                print(f"  => {total_tokens} tokens, {wall:.1f}s wall, {eff_tps:.1f} effective tok/s")

    # --- summary ---
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    # Baseline: concurrency=1
    c1_runs = [r for r in all_runs if r.concurrency == 1]
    if c1_runs:
        baseline_wall = sum(r.wall_seconds for r in c1_runs) / len(c1_runs)
        baseline_tps = sum(r.effective_tok_per_sec for r in c1_runs) / len(c1_runs)
    else:
        baseline_wall = 1
        baseline_tps = 1

    # Per-task-type baseline tok/s (serial, C=1)
    baseline_per_type = {}
    for tt in TASK_TYPES:
        vals = []
        for r in c1_runs:
            for t in r.tasks:
                if t.task_type == tt:
                    vals.append(t.tok_per_sec)
        baseline_per_type[tt] = sum(vals) / len(vals) if vals else 0

    print(f"\n{'Concurrency':<14} {'Avg Wall (s)':>13} {'Eff. tok/s':>12} {'Speedup':>9} {'Avg Tokens':>12}")
    print(f"{'-'*62}")

    for c in args.concurrency:
        c_runs = [r for r in all_runs if r.concurrency == c]
        if not c_runs:
            continue
        avg_wall = sum(r.wall_seconds for r in c_runs) / len(c_runs)
        avg_tps = sum(r.effective_tok_per_sec for r in c_runs) / len(c_runs)
        avg_tokens = sum(r.aggregate_tokens for r in c_runs) / len(c_runs)
        speedup = baseline_wall / avg_wall if avg_wall > 0 else 0
        print(f"{c:<14} {avg_wall:>13.1f} {avg_tps:>12.1f} {speedup:>8.2f}x {avg_tokens:>12.0f}")

    # Per-task-type contention across concurrency levels
    print(f"\nPer-task-type throughput (tok/s) — different problems, same complexity:")
    type_labels = {
        "algorithm": "Algorithm",
        "testing": "Testing",
        "refactoring": "Refactoring",
        "system_design": "System Design",
    }
    print(f"{'Task Type':<20s}", end="")
    for c in args.concurrency:
        print(f" {'C='+str(c):>8s}", end="")
    print(f" {'Contention':>11s}")
    print(f"{'-'*(20 + 8*len(args.concurrency) + 12)}")

    for tt in TASK_TYPES:
        print(f"{type_labels[tt]:<20s}", end="")
        per_c_tps = {}
        for c in args.concurrency:
            c_runs = [r for r in all_runs if r.concurrency == c]
            vals = []
            for r in c_runs:
                for t in r.tasks:
                    if t.task_type == tt:
                        vals.append(t.tok_per_sec)
            avg = sum(vals) / len(vals) if vals else 0
            per_c_tps[c] = avg
            print(f" {avg:>8.1f}", end="")

        max_c = max(args.concurrency)
        if baseline_per_type.get(tt, 0) > 0 and max_c in per_c_tps:
            contention = (1 - per_c_tps[max_c] / baseline_per_type[tt]) * 100
            print(f" {contention:>10.1f}%", end="")
        print()

    # Truncation report
    if truncations:
        print(f"\nTRUNCATIONS (hit max_tokens={args.max_tokens}):")
        for t in truncations:
            print(t)
    else:
        print(f"\nAll responses completed naturally (finish_reason: stop)")

    # --- CSV ---
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "concurrency", "run", "task", "label", "task_type",
            "completion_tokens", "wall_seconds", "tok_per_sec", "finish_reason",
            "run_wall_seconds", "run_aggregate_tokens", "run_effective_tok_per_sec",
        ])
        for r in all_runs:
            for t in r.tasks:
                writer.writerow([
                    r.concurrency, r.run_number, t.prompt_key, t.label, t.task_type,
                    t.completion_tokens, f"{t.wall_seconds:.3f}", f"{t.tok_per_sec:.1f}",
                    t.finish_reason, f"{r.wall_seconds:.3f}",
                    r.aggregate_tokens, f"{r.effective_tok_per_sec:.1f}",
                ])
    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
