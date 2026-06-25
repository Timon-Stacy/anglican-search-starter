"""Dynamic micro-batching for semantic search.

Concurrent queries are collected for a few milliseconds, then processed
together: the query embeddings, the FAISS search, and the cross-encoder rerank
all run as single *batched* GPU calls. On one GPU this is far higher throughput
than handling queries one at a time, because batched inference keeps the device
busy instead of paying per-call overhead per request.

A single worker thread owns all model/index/DB access, so there's no cross-thread
contention; request threads just enqueue a job and wait on an Event.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import DEFAULT_RERANK_POOL, QUERY_PREFIX
from .search import Filters, Searcher


@dataclass
class _Job:
    query: str
    k: int
    rerank: bool
    rerank_pool: int
    fetch_k: int
    filters: Filters
    event: threading.Event = field(default_factory=threading.Event)
    result: list[dict[str, Any]] | None = None
    error: BaseException | None = None


class BatchedSearch:
    def __init__(self, searcher: Searcher, max_batch: int = 16, max_wait_ms: float = 8.0):
        self.s = searcher
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._q: queue.Queue[_Job] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="search-batcher")
        self._worker.start()

    def search(self, query: str, k: int = 5, *, rerank: bool = True,
               filters: Filters | None = None, rerank_pool: int | None = None,
               timeout: float = 30.0) -> list[dict[str, Any]]:
        filters = filters or Filters()
        job = _Job(query=query, k=k, rerank=rerank,
                   rerank_pool=rerank_pool or DEFAULT_RERANK_POOL,
                   fetch_k=600 if filters.any() else 200, filters=filters)
        self._q.put(job)
        if not job.event.wait(timeout):
            raise TimeoutError("search timed out")
        if job.error is not None:
            raise job.error
        return job.result or []

    # -- worker thread ------------------------------------------------------
    def _collect(self) -> list[_Job]:
        batch = [self._q.get()]  # block for the first job
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(self._q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while True:
            batch = self._collect()
            try:
                self._process(batch)
            except BaseException as e:  # noqa: BLE001 - report to every waiter
                for j in batch:
                    if not j.event.is_set():
                        j.error = e
                        j.event.set()

    def _process(self, batch: list[_Job]) -> None:
        s = self.s
        # 1) one batched embedding call for all queries
        with s._gpu_lock:
            qv = s.model.encode([QUERY_PREFIX + j.query for j in batch],
                                normalize_embeddings=True,
                                convert_to_numpy=True).astype("float32")
        # 2) one batched FAISS search (efSearch tracks the largest k in the batch)
        maxfetch = max(max(j.fetch_k, j.k) for j in batch)
        s._ensure_ef(maxfetch)
        scores, ids = s.index.search(qv, maxfetch)

        # 3) per-job candidate fetch; gather rerank pairs for one batched predict
        states: list[tuple[_Job, dict, list[tuple[int, float]]]] = []
        flat_pairs: list[list[str]] = []
        slices: list[tuple[int, int, int, list[tuple[int, float]]]] = []
        for i, j in enumerate(batch):
            row_ids = [int(x) for x in ids[i] if x != -1][: j.fetch_k]
            rows = s._fetch(row_ids, j.filters)
            ordered = [(cid, float(sc)) for cid, sc in zip(row_ids, scores[i]) if cid in rows]
            states.append((j, rows, ordered))
            if j.rerank and ordered:
                pool = ordered[: j.rerank_pool]
                start = len(flat_pairs)
                flat_pairs.extend([j.query, rows[cid]["text"]] for cid, _ in pool)
                slices.append((i, start, len(flat_pairs), pool))

        # 4) one batched rerank across all jobs
        reranked: dict[int, list[tuple[int, float]]] = {}
        if flat_pairs:
            with s._gpu_lock:
                rr = [float(x) for x in s.reranker.predict(flat_pairs, convert_to_numpy=True)]
            for (idx, start, end, pool) in slices:
                sub = rr[start:end]
                order = sorted(range(len(pool)), key=lambda t: sub[t], reverse=True)
                reranked[idx] = [(pool[t][0], sub[t]) for t in order]

        # 5) assemble each result and release its waiter
        for i, (j, rows, ordered) in enumerate(states):
            final = reranked[i][: j.k] if i in reranked else ordered[: j.k]
            j.result = [s._format(rows[cid], sc) for cid, sc in final]
            j.event.set()
