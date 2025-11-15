import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple


"""
Minimal EXAONE×UZR external memory gateway (no external deps).

Implements endpoints:
- POST /mem/search  { q, k, filters?, lambda? } -> candidates
- POST /mem/write   { ops: [UPSERT_NODE|ADD_EDGE...], author, justification, sig? } -> ack or 409
- GET  /mem/stream  Server-Sent Events (SSE): write/read events and heartbeats

Storage layout under store_dir (default: ./logus-exaone):
- mem_events.jsonl    L1 append-only event log with hash chain (blake2b)
- l2_graph.json       L2 symbolic graph (nodes/edges, with etag/version)
- shadow_bank.json    Low-trust items parked here (trust < 0.6)

Notes/limits:
- Ed25519 signature is accepted as a field; verification is not performed (sig_status='unverified').
- ANN is a light cosine search on toy embeddings derived deterministically from text.
- ETag/version conflict returns 409 with a shallow diff for client-side patching.
"""


# -----------------------------
# Small utilities
# -----------------------------

def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def l2_norm(vec: List[float]) -> List[float]:
    s = sum(v * v for v in vec) ** 0.5
    if s <= 1e-12:
        return vec
    return [v / s for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(num / (da * db))


def toy_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic, dependency-free text embedding (toy).

    - Tokenizes on whitespace
    - Each token is used to seed a local PRNG (via blake2b) and generate dim values
    - Sums across tokens and L2-normalizes
    """
    import hashlib
    import random as pyrand

    acc = [0.0] * dim
    for tok in (text or "").split():
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h, "little")
        rng = pyrand.Random(seed)
        for i in range(dim):
            # zero-mean unit-ish variance draws
            acc[i] += rng.uniform(-1.0, 1.0)
    return l2_norm(acc)


def blake2b_hex(data: bytes) -> str:
    import hashlib
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def shallow_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    keys = set(a.keys()) | set(b.keys())
    out: Dict[str, Tuple[Any, Any]] = {}
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            out[k] = (va, vb)
    return out


# -----------------------------
# Event bus for SSE
# -----------------------------


class EventBus:
    def __init__(self) -> None:
        from queue import Queue
        self._subs: List["Queue[str]"] = []
        self._lock = threading.Lock()

    def publish(self, data: Dict[str, Any]) -> None:
        from queue import Queue
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass

    def subscribe(self):
        from queue import Queue
        q: "Queue[str]" = Queue(maxsize=1024)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass


# -----------------------------
# Optional UZR in-process mirror (real-time L2→UZR memory sync)
# -----------------------------

class UzrMirror:
    """Maintain an in-process UZR model+memory and mirror L2 node writes.

    Enable via environment:
      UZR_CKPT=/path/to/ckpt.pt  (required)
      UZR_DEVICE=cpu|cuda        (optional, default 'cpu')
    """

    def __init__(self, store: "MemStore") -> None:
        self.enabled = False
        self._model = None
        self._tok = None
        self._mem = None
        self._step = 0
        self._store = store
        self._error: Optional[str] = None
        ckpt = os.environ.get("UZR_CKPT", "").strip()
        # 자동 탐색: 환경변수 없으면 대표 파일명 시도
        if not ckpt:
            for cand in (
                "uzr_3brains_ckpt_best.pt",
                "uzr_3brains_ckpt_last.pt",
                "uzr_3brains_ckpt.pt",
            ):
                if os.path.exists(cand):
                    ckpt = cand
                    break
        if not ckpt:
            self._error = "no_ckpt"
            return
        self._device = os.environ.get("UZR_DEVICE", "cpu")
        try:
            import torch as _torch
            from model import UZRModel, ByteTokenizer, KoEnTokenizer  # type: ignore
            from memory import CompressedMemory  # type: ignore
            data = _torch.load(ckpt, map_location="cpu", weights_only=False)
            args = data.get("args", {}) if isinstance(data, dict) else {}
            rdw = data.get("model", {}).get("readout.weight") if isinstance(data, dict) else None
            rows = rdw.size(0) if hasattr(rdw, "size") else None
            max_len = int(args.get("max_len", 128))
            tok = ByteTokenizer(max_len=max_len) if rows == 258 else KoEnTokenizer(max_len=max_len)
            mem = None
            if isinstance(data, dict) and "memory" in data:
                mem = CompressedMemory(max_items=32000, device=self._device, enable_learning=True, learn_hidden=512, learn_depth=3, warmup_steps=1)
                try:
                    mem.load_state_dict(data["memory"])  # reuse if present
                    mem.enable_learning = True
                    mem.warmup_steps = 1
                    if hasattr(mem, "write_threshold_warmup_end"):
                        mem.write_threshold_warmup_end = 1
                except Exception:
                    pass
            if mem is None:
                mem = CompressedMemory(max_items=32000, device=self._device, enable_learning=True, learn_hidden=512, learn_depth=3, warmup_steps=1)
            model = UZRModel(
                vocab_size=tok.vocab_size,
                d_model=args.get("d_model", 256),
                z_dim=args.get("z_dim", 128),
                max_len=max_len,
                z_think_dim=args.get("z_think_dim", 64),
                z_lang_dim=args.get("z_lang_dim", 32),
                num_langs=args.get("num_langs", 4),
                identity_self_dim=args.get("identity_self_dim", 32),
                identity_intent_dim=args.get("identity_intent_dim"),
                memory=mem,
            )
            sd = data["model"] if isinstance(data, dict) else data
            model.load_state_dict(sd)
            model = model.to(self._device)
            model.eval()
            self._model = model
            self._tok = tok
            self._mem = mem
            self.enabled = True
            print(f"[mem][uzr] mirror enabled (ckpt={os.path.basename(ckpt)}, device={self._device})")
        except Exception as e:
            print(f"[mem][uzr] mirror disabled: {e}")
            self._error = str(e)
            self.enabled = False

    def stats(self) -> Dict[str, Any]:
        if not self.enabled:
            base = {"enabled": False}
            if self._error:
                base["error"] = self._error
            return base
        try:
            return {
                "enabled": True,
                "items": len(self._mem.items) if self._mem is not None else 0,
                "step": int(self._step),
                "max_items": int(getattr(self._mem, "max_items", 0)),
            }
        except Exception:
            return {"enabled": True}

    def ingest_node(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self._model or not self._tok or not self._mem:
            return None
        try:
            import torch as _torch
            from memory import make_sketch  # type: ignore
            title = (node.get("title", "") or "").strip()
            body = (node.get("body", "") or "").strip()
            text = (title + "\n" + body).strip()
            if not text:
                return None
            X = self._tok.encode(text).unsqueeze(0).to(self._model.readout.weight.device)
            with _torch.no_grad():
                avg_emb = self._model.avg_embed(X)[0].detach().cpu()
            z_pred = None
            try:
                z_pred = self._mem.predict(avg_emb, topk=4, blend=0.5)
            except Exception:
                z_pred = None
            if z_pred is None:
                z_pred = self._model.init_z(batch_size=1)[0].detach().cpu() * 0.0
            meta = {
                "node_id": node.get("id"),
                "type": node.get("type"),
                "tags": node.get("tags", []),
                "trust": float(node.get("trust", 0.5)),
                "parents": node.get("parents", []),
                "proj": MemStore._node_project(self._store, node),
            }
            key, val = make_sketch(avg_emb, z_pred, meta=meta, repr_text=text)
            self._step += 1
            if hasattr(self._mem, "add_with_policy"):
                # Type-weighted bench: Decision/Preference는 항상 커밋, Fact는 기본 정책, Episode는 보수적
                ntype = (node.get("type") or "").lower()
                if ntype in {"decision", "preference"}:
                    bench = (lambda: True)
                elif ntype in {"episode"}:
                    # episode는 과다 연결 방지를 위해 약하게 제한(신뢰도 기준)
                    bench = (lambda: float(node.get("trust", 0.5)) >= 0.6)
                else:
                    bench = None
                mem_meta = {"luria_intent_force": self._intent_force_from_model()}
                return self._mem.add_with_policy(key=key, val=val, step=self._step, meta=mem_meta, bench_callback=bench)
            self._mem.add(key, val, step=self._step)
            return {"status": "accepted"}
        except Exception as e:
            try:
                print(f"[mem][uzr] ingest error: {e}")
            except Exception:
                pass
            return None

    def backfill(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Backfill existing L2 nodes into UZR memory (best-effort).

        Args:
            limit: max nodes to process (None → all)
        Returns:
            {processed, accepted}
        """
        processed = 0
        accepted = 0
        if not self._store:
            return {"processed": processed, "accepted": accepted}
        # Order by updated_ts if possible
        items = list(self._store.nodes.values())
        try:
            items.sort(key=lambda n: n.get("updated_ts", ""))
        except Exception:
            pass
        for n in items[: (len(items) if limit is None else max(0, int(limit))) ]:
            processed += 1
            dec = self.ingest_node(n)
            if isinstance(dec, dict) and dec.get("status") in {"accepted", "merged"}:
                accepted += 1
        return {"processed": processed, "accepted": accepted}


    def _intent_force_from_model(self) -> Optional[bool]:
        if getattr(self, "_model", None) is None:
            return None
        try:
            _, toggle = self._model.identity_intent_control()
        except Exception:
            return None
        if toggle <= -0.5:
            return False
        if toggle >= 0.5:
            return True
        return None


class UzrSync(threading.Thread):
    """Subscribe to event bus and mirror L2 node UPSERTs to UZR memory."""

    def __init__(self, store: "MemStore", bus: EventBus):
        super().__init__(daemon=True)
        self.store = store
        self.bus = bus
        self.mirror = UzrMirror(store)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        # 구독은 항상 수행(요구 사항). 미러 비활성일 땐 이벤트만 소비.
        q = self.bus.subscribe()
        try:
            while not self._stop.is_set():
                try:
                    payload = q.get(timeout=1.0)
                except Exception:
                    continue
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue
                if ev.get("ev") == "WRITE_OK":
                    det = ev.get("detail") or {}
                    results = det.get("results") if isinstance(det, dict) else None
                    if not isinstance(results, list):
                        continue
                    for r in results:
                        if isinstance(r, dict) and r.get("id"):
                            nid = r.get("id")
                            node = self.store.nodes.get(nid) or self.store.shadow.get(nid)
                            if node and self.mirror.enabled:
                                dec = self.mirror.ingest_node(node)
                                try:
                                    MemHandler.bus.publish({
                                        "ts": utc_ts(),
                                        "ev": "UZR_MIRROR",
                                        "id": nid,
                                        "status": (dec or {}).get("status", "noop"),
                                        "reason": (dec or {}).get("reason")
                                    })
                                except Exception:
                                    pass
        finally:
            self.bus.unsubscribe(q)


# -----------------------------
# L1/L2 store
# -----------------------------


@dataclass
class Node:
    id: str
    type: str
    title: str = ""
    body: str = ""
    tags: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    trust: float = 0.5
    version: str = "v1"
    prev_version: Optional[str] = None
    etag: str = "v1"
    updated_ts: str = field(default_factory=utc_ts)
    embedding: Optional[List[float]] = None
    shadow: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "parents": self.parents,
            "trust": self.trust,
            "version": self.version,
            "prev_version": self.prev_version,
            "etag": self.etag,
            "updated_ts": self.updated_ts,
            "embedding": self.embedding,
            "shadow": self.shadow,
        }


class MemStore:
    """Local JSON-backed store for L1/L2 with minimal locking."""

    def __init__(self, store_dir: str = "logus-exaone") -> None:
        self.store_dir = store_dir
        ensure_dir(self.store_dir)

        self.l1_path = os.path.join(self.store_dir, "mem_events.jsonl")
        self.l2_path = os.path.join(self.store_dir, "l2_graph.json")
        self.shadow_path = os.path.join(self.store_dir, "shadow_bank.json")
        self._lock = threading.RLock()
        self._last_hash_path = os.path.join(self.store_dir, "last_hash.txt")

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._recent: deque = deque(maxlen=1000)  # recent events
        self._last_hash: Optional[str] = None
        self.last_event_unix: float = time.time()

        # UZR autonomous policy & state
        self.policy: Dict[str, Any] = {
            "dup_skip_thr": 0.97,
            "near_merge_thr": 0.90,
            "trust_shadow_thr": 0.60,
            "writes_per_min": 120,
            "two_pc_ttl": 300,
            "stage_max_age": 7 * 24 * 3600,
            "promote_batch": 8,
        }
        self._rate_window: Dict[str, Tuple[int, int]] = {}
        self._staging: Dict[str, Dict[str, Any]] = {}
        self.shadow: Dict[str, Dict[str, Any]] = {}

        self._load_all()

    # ---------- load/save ----------
    def _load_all(self) -> None:
        with self._lock:
            # L2 graph
            if os.path.exists(self.l2_path):
                try:
                    with open(self.l2_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.nodes = data.get("nodes", {})
                        self.edges = data.get("edges", [])
                except Exception:
                    self.nodes, self.edges = {}, []
            # Shadow bank (if present)
            if os.path.exists(self.shadow_path):
                try:
                    with open(self.shadow_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                        self.shadow = s.get("nodes", {})
                except Exception:
                    self.shadow = {}
            # last hash
            if os.path.exists(self._last_hash_path):
                try:
                    with open(self._last_hash_path, "r", encoding="utf-8") as f:
                        self._last_hash = f.read().strip() or None
                except Exception:
                    self._last_hash = None

    def _save_l2(self) -> None:
        with self._lock:
            data = {
                "version": 1,
                "last_updated": utc_ts(),
                "nodes": self.nodes,
                "edges": self.edges,
            }
            tmp = self.l2_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.l2_path)

    def _save_shadow(self) -> None:
        with self._lock:
            data = {
                "version": 1,
                "last_updated": utc_ts(),
                "nodes": self.shadow,
            }
            tmp = self.shadow_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.shadow_path)

    # ---------- L1 logging ----------
    def _append_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            event = dict(event)
            event.setdefault("ts", utc_ts())
            # compute hash chain
            body = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            prev = self._last_hash or ""
            chain_input = prev.encode("utf-8") + b"\n" + body
            ev_hash = blake2b_hex(chain_input)
            event["prev_hash"] = prev or None
            event["hash"] = ev_hash

            # append jsonl
            with open(self.l1_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._last_hash = ev_hash
            try:
                with open(self._last_hash_path, "w", encoding="utf-8") as f:
                    f.write(ev_hash)
            except Exception:
                pass
            # cache
            self._recent.append(event)
            # update last event time
            self.last_event_unix = time.time()
            return event

    # ---------- search ----------
    def _node_project(self, node: Dict[str, Any]) -> Optional[str]:
        # from parents like "proj:uzr" or tags
        for p in node.get("parents", []) or []:
            if isinstance(p, str) and p.startswith("proj:"):
                return p.split(":", 1)[1]
        for t in node.get("tags", []) or []:
            if t.startswith("proj:"):
                return t.split(":", 1)[1]
        return None

    def search(self, q: str, k: int = 8, filters: Optional[Dict[str, Any]] = None,
               type_lambda: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        now = time.time()
        qv = toy_embed(q or "")
        filters = filters or {}
        types: Optional[List[str]] = filters.get("type")
        project: Optional[List[str]] = filters.get("project")
        tags_req: Optional[List[str]] = filters.get("tags")
        include_shadow: bool = bool(filters.get("include_shadow", False))

        # collect candidates
        cands: List[Tuple[float, Dict[str, Any]]] = []
        for node in self.nodes.values():
            if node.get("shadow") and not include_shadow:
                continue
            if types and node.get("type") not in types:
                continue
            if tags_req:
                ntags = set(node.get("tags", []) or [])
                if not ntags.issuperset(set(tags_req)):
                    continue
            if project:
                p = self._node_project(node)
                if p not in set(project):
                    continue
            emb = node.get("embedding") or toy_embed(f"{node.get('title','')} {node.get('body','')}")
            node["embedding"] = emb  # cache
            sim = cosine(qv, emb)
            trust = float(node.get("trust", 0.5))
            # recency/staleness
            try:
                uts = node.get("updated_ts")
                t_struct = time.strptime(uts, "%Y-%m-%dT%H:%M:%SZ") if uts else None
                age = max(0.0, now - time.mktime(t_struct)) if t_struct else 0.0
            except Exception:
                age = 0.0
            staleness_pen = min(0.25, age / (7 * 24 * 3600) * 0.05)  # up to -0.25 over 5+ weeks
            base = sim + 0.2 * trust - staleness_pen
            if type_lambda and node.get("type") in type_lambda:
                base *= float(type_lambda[node["type"]])
            cands.append((base, node))

        cands.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, n in cands[: max(1, int(k))]:
            out.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "title": n.get("title", ""),
                "snippet": (n.get("body", "") or "")[:240],
                "score": round(float(score), 6),
                "trust": n.get("trust", 0.5),
                "tags": n.get("tags", []),
                "parents": n.get("parents", []),
                "etag": n.get("etag"),
            })
        # log READ event (compact)
        self._append_event({
            "ev": "READ",
            "path": "/mem/search",
            "q": q,
            "k": k,
            "filters": filters,
            "result_cnt": len(out),
        })
        return out

    # ---------- write (MAL) ----------
    # Helpers for novelty, rate limiting, and 2PC
    def _nearest_sim(self, emb: List[float]) -> float:
        best = -1.0
        for n in self.nodes.values():
            e = n.get("embedding") or toy_embed(f"{n.get('title','')} {n.get('body','')}")
            n["embedding"] = e
            sim = cosine(emb, e)
            if sim > best:
                best = sim
        return best

    def _k3_avg(self, emb: List[float]) -> float:
        sims: List[float] = []
        for n in self.nodes.values():
            e = n.get("embedding") or toy_embed(f"{n.get('title','')} {n.get('body','')}")
            n["embedding"] = e
            sims.append(cosine(emb, e))
        if not sims:
            return -1.0
        sims.sort(reverse=True)
        k = min(3, len(sims))
        return sum(sims[:k]) / float(k)

    def _rate_limit_ok(self, author: Optional[str]) -> bool:
        if not author:
            return True
        per_min = int(self.policy.get("writes_per_min", 120))
        now_min = int(time.time() // 60)
        win, cnt = self._rate_window.get(author, (now_min, 0))
        if win != now_min:
            win, cnt = now_min, 0
        if cnt >= per_min:
            return False
        self._rate_window[author] = (win, cnt + 1)
        return True
    def _next_version(self, cur: Optional[str]) -> Tuple[str, Optional[str]]:
        if not cur:
            return "v1", None
        try:
            n = int(cur.lstrip("v")) + 1
        except Exception:
            n = 1
        return f"v{n}", cur

    def _apply_upsert_node(self, op: Dict[str, Any], author: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # returns (ack, conflict)
        nid = op.get("id")
        if not nid:
            return None, {"error": "missing node id"}

        body_fields = {k: op.get(k) for k in ("type", "title", "body", "tags", "parents", "trust") if k in op}
        req_etag = op.get("etag")
        precond = op.get("precond") or {}

        # enforce simple precond.absent: ["id:...", ...]
        absent = precond.get("absent") or []
        for cond in absent:
            if isinstance(cond, str) and cond.startswith("id:"):
                cid = cond.split(":", 1)[1]
                if cid in self.nodes:
                    return None, {"status": 409, "reason": f"precond.absent failed: {cid}"}

        # Project whitelist enforcement (optional)
        proj_allow = set()
        try:
            env = os.environ.get("MEM_PROJECT_ALLOW", "").strip()
            if env:
                proj_allow = {x.strip() for x in env.split(",") if x.strip()}
        except Exception:
            proj_allow = set()

        existing = self.nodes.get(nid) or self.shadow.get(nid)
        if existing is None:
            # create
            n = Node(id=nid, type=body_fields.get("type") or op.get("type") or "Fact")
            n.title = body_fields.get("title") or ""
            n.body = body_fields.get("body") or ""
            n.tags = list(body_fields.get("tags") or [])
            n.parents = list(body_fields.get("parents") or [])
            n.trust = float(body_fields.get("trust", 0.5))
            n.embedding = toy_embed(f"{n.title} {n.body}")
            # project whitelist
            proj = self._node_project(n.to_dict())
            if proj_allow and proj not in proj_allow:
                return None, {"status": 403, "error": f"project_not_allowed:{proj}"}
            # novelty gating
            sim_max = self._nearest_sim(n.embedding)
            k3 = self._k3_avg(n.embedding)
            dup_thr = float(self.policy.get("dup_skip_thr", 0.98))
            near_thr = float(self.policy.get("near_merge_thr", 0.90))
            trust_shadow = float(self.policy.get("trust_shadow_thr", 0.60))
            if sim_max >= dup_thr:
                ev = self._append_event({
                    "ev": "WRITE_SKIP",
                    "op": "UPSERT_NODE",
                    "id": nid,
                    "reason": "dup_skip",
                    "sim_max": sim_max,
                    "author": author,
                })
                return {"id": nid, "action": "skipped", "reason": "dup_skip", "event": ev}, None
            # decide destination
            action = "accepted"
            target = self.nodes
            if (k3 >= near_thr) or (n.trust < trust_shadow):
                n.shadow = True
                target = self.shadow
                action = "staged"
            else:
                n.shadow = False
            target[nid] = n.to_dict()
            if n.shadow:
                self._save_shadow()
            else:
                self._save_l2()
            ev = self._append_event({
                "ev": "WRITE_OK",
                "op": "UPSERT_NODE",
                "id": nid,
                "version": target[nid].get("version"),
                "etag": target[nid].get("etag"),
                "author": author,
                "action": action,
                "sim_max": sim_max,
                "k3_avg": k3,
                "sig_status": "unverified",
            })
            return {"id": nid, "version": target[nid]["version"], "etag": target[nid]["etag"], "action": action, "event": ev}, None
        # update path: conflict check
        if req_etag and req_etag != existing.get("etag"):
            return None, {
                "status": 409,
                "id": nid,
                "server_etag": existing.get("etag"),
                "diff": shallow_diff({k: existing.get(k) for k in body_fields.keys()}, body_fields),
            }

        # update path: project whitelist
        proj2 = self._node_project(existing)
        if proj_allow and proj2 not in proj_allow:
            return None, {"status": 403, "error": f"project_not_allowed:{proj2}"}

        newver, prev = self._next_version(existing.get("version"))
        existing["prev_version"] = prev
        existing["version"] = newver
        existing["etag"] = newver
        existing["updated_ts"] = utc_ts()
        # apply fields
        for k, v in body_fields.items():
            if v is not None:
                existing[k] = v if k not in ("tags", "parents") else list(v)
        # refresh embedding if title/body changed
        if any(k in body_fields for k in ("title", "body")):
            existing["embedding"] = toy_embed(f"{existing.get('title','')} {existing.get('body','')}")
        # shadow routing based on trust
        existing["shadow"] = bool(float(existing.get("trust", 0.5)) < float(self.policy.get("trust_shadow_thr", 0.60)))
        if existing["shadow"]:
            self.shadow[nid] = existing
            self.nodes.pop(nid, None)
            self._save_shadow()
        else:
            self.nodes[nid] = existing
            self.shadow.pop(nid, None)
            self._save_l2()
        ev = self._append_event({
            "ev": "WRITE_OK",
            "op": "UPSERT_NODE",
            "id": nid,
            "version": existing.get("version"),
            "etag": existing.get("etag"),
            "author": author,
            "action": "updated",
            "sig_status": "unverified",
        })
        return {"id": nid, "version": existing["version"], "etag": existing["etag"], "event": ev}, None

    def _apply_add_edge(self, op: Dict[str, Any], author: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        src = op.get("src"); dst = op.get("dst"); rel = op.get("rel")
        if not (src and dst and rel):
            return None, {"error": "missing src/dst/rel"}
        w = float(op.get("weight", 1.0))
        edge = {"src": src, "dst": dst, "rel": rel, "weight": w}
        # dedup simple
        for e in self.edges:
            if e.get("src") == src and e.get("dst") == dst and e.get("rel") == rel:
                e["weight"] = w
                self._save_l2()
                ev = self._append_event({"ev": "WRITE_OK", "op": "ADD_EDGE", **edge, "author": author})
                return {"edge": edge, "event": ev}, None
        self.edges.append(edge)
        self._save_l2()
        ev = self._append_event({"ev": "WRITE_OK", "op": "ADD_EDGE", **edge, "author": author})
        return {"edge": edge, "event": ev}, None

    def write(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        author = payload.get("author")
        just = payload.get("justification")
        sig = payload.get("sig")
        ops = payload.get("ops") or []
        # ops limit
        try:
            max_ops = int(os.environ.get("MEM_MAX_OPS", "256"))
        except Exception:
            max_ops = 256
        if not isinstance(ops, list) or len(ops) > max_ops:
            ev = self._append_event({"ev": "ERROR", "path": "/mem/write", "error": "too_many_ops", "author": author})
            return {}, {"status": 400, "error": "too_many_ops", "event": ev}
        two_pc = bool(payload.get("two_pc", False))

        # rate limit per author
        if not self._rate_limit_ok(author):
            ev = self._append_event({"ev": "RATE_LIMIT", "path": "/mem/write", "author": author})
            return {}, {"status": 429, "error": "rate_limit", "event": ev}

        # 2PC prepare
        if two_pc:
            ticket = f"t:{int(time.time())}-{abs(hash(json.dumps(ops, ensure_ascii=False)))%1000000}"
            self._staging[ticket] = {"payload": payload, "author": author, "ts": time.time()}
            self._append_event({"ev": "PREPARED", "ticket": ticket, "author": author, "ops_cnt": len(ops)})
            return {"status": "prepared", "ticket": ticket}, None

        ack: Dict[str, Any] = {"results": []}

        for op in ops:
            otype = op.get("op")
            if otype == "UPSERT_NODE":
                res, conflict = self._apply_upsert_node(op, author=author)
                if conflict:
                    # log conflict
                    ev = self._append_event({
                        "ev": "CONFLICT",
                        "op": "UPSERT_NODE",
                        "id": op.get("id"),
                        "conflict": conflict,
                        "author": author,
                    })
                    return {}, {"status": 409, **conflict, "event": ev}
                ack["results"].append(res)
            elif otype == "ADD_EDGE":
                res, conflict = self._apply_add_edge(op, author=author)
                if conflict:
                    ev = self._append_event({
                        "ev": "ERROR",
                        "op": "ADD_EDGE",
                        "error": conflict,
                        "author": author,
                    })
                    return {}, {"status": 400, **conflict, "event": ev}
                ack["results"].append(res)
            else:
                ev = self._append_event({"ev": "ERROR", "op": otype or "", "error": "unsupported op", "author": author})
                return {}, {"status": 400, "error": f"unsupported op: {otype}", "event": ev}

        # envelope-level log
        self._append_event({
            "ev": "ENVELOPE",
            "path": "/mem/write",
            "author": author,
            "justification": just,
            "sig_present": bool(sig),
            "sig_status": "unverified",
            "ops_cnt": len(ops),
        })
        return ack, None

    def commit_ticket(self, ticket: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        ent = self._staging.get(ticket)
        if not ent:
            return {}, {"status": 404, "error": "unknown_ticket"}
        ttl = float(self.policy.get("two_pc_ttl", 300))
        if time.time() - float(ent.get("ts", 0)) > ttl:
            self._staging.pop(ticket, None)
            return {}, {"status": 410, "error": "ticket_expired"}
        payload = dict(ent.get("payload") or {})
        payload.pop("two_pc", None)
        ack, conflict = self.write(payload)
        if not conflict:
            self._staging.pop(ticket, None)
        return ack, conflict

    def rebalance(self) -> Dict[str, Any]:
        promoted = 0
        examined = 0
        batch = int(self.policy.get("promote_batch", 8))
        now = time.time()
        try:
            import calendar
        except Exception:
            calendar = None
        # iterate oldest-first
        items = sorted(self.shadow.items(), key=lambda kv: kv[1].get("updated_ts", ""))
        for nid, n in items:
            examined += 1
            age = 0
            if calendar:
                try:
                    t_struct = time.strptime(n.get("updated_ts"), "%Y-%m-%dT%H:%M:%SZ")
                    age = now - calendar.timegm(t_struct)
                except Exception:
                    age = 0
            cond_age = age >= float(self.policy.get("stage_max_age", 7 * 24 * 3600))
            cond_trust = float(n.get("trust", 0.5)) >= (float(self.policy.get("trust_shadow_thr", 0.60)) + 0.1)
            if cond_age or cond_trust:
                n["shadow"] = False
                self.nodes[nid] = n
                self.shadow.pop(nid, None)
                promoted += 1
                self._append_event({"ev": "PROMOTE", "id": nid})
                if promoted >= batch:
                    break
        if promoted:
            self._save_shadow()
            self._save_l2()
        return {"promoted": promoted, "examined": examined, "shadow_size": len(self.shadow), "mem_size": len(self.nodes)}


# -----------------------------
# HTTP handler
# -----------------------------


class MemHandler(BaseHTTPRequestHandler):
    store = MemStore()
    bus = EventBus()
    uzr_sync: Optional[UzrSync] = None

    server_version = "EXAONE-UZR-Mem/0.1"
    # SSE connection accounting
    sse_clients: int = 0
    sse_by_ip: Dict[str, int] = {}

    def _cors_origin(self) -> str:
        origin = os.environ.get("MEM_CORS_ORIGIN", "*")
        return origin

    def _set_headers(self, status: int = 200, ctype: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.send_header("Access-Control-Allow-Headers", "*, content-type, authorization, x-signature, x-api-key")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def _audit(self, ev: Dict[str, Any]) -> None:
        try:
            base = getattr(MemHandler.store, "store_dir", ".")
            ensure_dir(base)
            path = os.path.join(base, "security_audit.jsonl")
            rec = dict(ev)
            rec.setdefault("ts", utc_ts())
            rec.setdefault("ip", self.client_address[0] if hasattr(self, "client_address") else None)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def do_OPTIONS(self):  # CORS preflight
        self._set_headers(HTTPStatus.NO_CONTENT)

    def do_GET(self):
        # IP allowlist for reads (optional)
        ip = self.client_address[0]
        allow = os.environ.get("MEM_IP_ALLOW_READ", "").strip()
        if allow:
            allow_set = {a.strip() for a in allow.split(",") if a.strip()}
            if ip not in allow_set:
                self._set_headers(HTTPStatus.FORBIDDEN)
                self.wfile.write(b"{\n \"error\": \"forbidden\" \n}")
                self._audit({"ev": "DENY", "method": "GET", "path": self.path, "reason": "ip_not_allowed"})
                return
        if self.path.startswith("/mem/stream"):
            self._handle_stream()
            return
        if self.path.startswith("/health"):
            self._set_headers(HTTPStatus.OK)
            self.wfile.write(b"{\n \"ok\": true \n}")
            return
        if self.path.startswith("/mem/uzr_stats"):
            stats = MemHandler.uzr_sync.mirror.stats() if (MemHandler.uzr_sync and MemHandler.uzr_sync.mirror) else {"enabled": False}
            self._set_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps(stats, ensure_ascii=False).encode("utf-8"))
            return
        if self.path.startswith("/mem/rebalance"):
            res = MemHandler.store.rebalance()
            self._set_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            MemHandler.bus.publish({"ts": utc_ts(), "ev": "REBALANCE", **res})
            return
        self._set_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(b"{\n \"error\": \"not found\" \n}")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        # Request size limit
        try:
            max_body = int(os.environ.get("MEM_MAX_BODY_BYTES", str(1024*1024)))
        except Exception:
            max_body = 1024 * 1024
        if length > max_body:
            self._set_headers(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            self.wfile.write(json.dumps({"error": "payload_too_large", "limit": max_body}).encode("utf-8"))
            return
        try:
            body = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            self._set_headers(HTTPStatus.BAD_REQUEST)
            self.wfile.write(b"{\n \"error\": \"invalid json\" \n}")
            return

        def _auth_required(write=False) -> bool:
            # IP allowlist
            ip = self.client_address[0]
            allow_env = os.environ.get("MEM_IP_ALLOW_WRITE" if write else "MEM_IP_ALLOW_READ", "").strip()
            if allow_env:
                allow_set = {a.strip() for a in allow_env.split(",") if a.strip()}
                if ip not in allow_set:
                    return False
            # API key / Bearer
            hdr = (self.headers.get("Authorization") or "").strip()
            api = (self.headers.get("X-API-Key") or "").strip()
            want = os.environ.get("MEM_WRITE_KEY" if write else "MEM_READ_KEY", "").strip()
            if want:
                ok = False
                if hdr.lower().startswith("bearer ") and hdr.split(" ",1)[1] == want:
                    ok = True
                if api and api == want:
                    ok = True
                if not ok:
                    return False
            # HMAC signature (optional)
            sig_key = os.environ.get("MEM_HMAC_KEY", "").encode("utf-8") if os.environ.get("MEM_HMAC_KEY") else None
            if sig_key:
                try:
                    import hmac, hashlib
                    given = (self.headers.get("X-Signature") or "").strip()
                    if not given:
                        return False
                    calc = hmac.new(sig_key, body, hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(calc, given):
                        return False
                except Exception:
                    return False
            return True

        if self.path.startswith("/mem/search"):
            if not _auth_required(write=False):
                self._set_headers(HTTPStatus.UNAUTHORIZED)
                self.wfile.write(b"{\n \"error\": \"unauthorized\" \n}")
                self._audit({"ev": "DENY", "method": "POST", "path": "/mem/search", "reason": "auth_failed"})
                return
            # require project filter when configured
            if os.environ.get("MEM_PROJECT_REQUIRE", "").strip().lower() in {"1","true","on","yes"}:
                projs = []
                try:
                    projs = ((payload.get("filters") or {}).get("project") or [])
                except Exception:
                    projs = []
                if not projs:
                    self._set_headers(HTTPStatus.BAD_REQUEST)
                    self.wfile.write(b"{\n \"error\": \"project_required\" \n}")
                    self._audit({"ev": "DENY", "method": "POST", "path": "/mem/search", "reason": "project_required"})
                    return
                # enforce allowlist if present
                allow = {a.strip() for a in (os.environ.get("MEM_PROJECT_ALLOW", "").split(",") if os.environ.get("MEM_PROJECT_ALLOW") else []) if a.strip()}
                if allow and any(p not in allow for p in projs):
                    self._set_headers(HTTPStatus.FORBIDDEN)
                    self.wfile.write(b"{\n \"error\": \"project_forbidden\" \n}")
                    self._audit({"ev": "DENY", "method": "POST", "path": "/mem/search", "reason": "project_forbidden"})
                    return
            q = payload.get("q", "")
            k = int(payload.get("k", 8))
            filters = payload.get("filters")
            lam = payload.get("lambda")
            res = self.store.search(q=q, k=k, filters=filters, type_lambda=lam)
            self._set_headers(HTTPStatus.OK)
            self.wfile.write(json.dumps({"candidates": res}, ensure_ascii=False).encode("utf-8"))
            # publish event
            MemHandler.bus.publish({"ts": utc_ts(), "ev": "READ", "path": "/mem/search", "k": k})
            self._audit({"ev": "READ", "status": "ok", "path": "/mem/search", "k": k})
            return

        if self.path.startswith("/mem/write"):
            if not _auth_required(write=True):
                self._set_headers(HTTPStatus.UNAUTHORIZED)
                self.wfile.write(b"{\n \"error\": \"unauthorized\" \n}")
                self._audit({"ev": "DENY", "method": "POST", "path": "/mem/write", "reason": "auth_failed"})
                return
            ack, conflict = self.store.write(payload)
            if conflict:
                self._set_headers(HTTPStatus.CONFLICT)
                self.wfile.write(json.dumps(conflict, ensure_ascii=False).encode("utf-8"))
                MemHandler.bus.publish({"ts": utc_ts(), "ev": "CONFLICT", "detail": conflict})
                self._audit({"ev": "WRITE", "status": "conflict", "detail": {"path": "/mem/write"}})
            else:
                self._set_headers(HTTPStatus.OK)
                self.wfile.write(json.dumps(ack, ensure_ascii=False).encode("utf-8"))
                MemHandler.bus.publish({"ts": utc_ts(), "ev": "WRITE_OK", "detail": ack})
                self._audit({"ev": "WRITE", "status": "ok", "detail": {"path": "/mem/write"}})
            return

        if self.path.startswith("/mem/backfill"):
            if not _auth_required(write=True):
                self._set_headers(HTTPStatus.UNAUTHORIZED)
                self.wfile.write(b"{\n \"error\": \"unauthorized\" \n}")
                self._audit({"ev": "DENY", "method": "POST", "path": "/mem/backfill", "reason": "auth_failed"})
                return
            limit = None
            try:
                limit = payload.get("limit")
            except Exception:
                pass
            if MemHandler.uzr_sync and MemHandler.uzr_sync.mirror and MemHandler.uzr_sync.mirror.enabled:
                res = MemHandler.uzr_sync.mirror.backfill(limit)
                self._set_headers(HTTPStatus.OK)
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            else:
                self._set_headers(HTTPStatus.SERVICE_UNAVAILABLE)
                self.wfile.write(json.dumps({"error": "uzr_mirror_disabled"}, ensure_ascii=False).encode("utf-8"))
            return

        if self.path.startswith("/mem/prepare"):
            if not _auth_required(write=True):
                self._set_headers(HTTPStatus.UNAUTHORIZED)
                self.wfile.write(b"{\n \"error\": \"unauthorized\" \n}")
                self._audit({"ev": "DENY", "method": "POST", "path": "/mem/prepare", "reason": "auth_failed"})
                return
            payload.setdefault("two_pc", True)
            ack, conflict = self.store.write(payload)
            if conflict:
                self._set_headers(HTTPStatus.CONFLICT)
                self.wfile.write(json.dumps(conflict, ensure_ascii=False).encode("utf-8"))
            else:
                self._set_headers(HTTPStatus.OK)
                self.wfile.write(json.dumps(ack, ensure_ascii=False).encode("utf-8"))
            return

        if self.path.startswith("/mem/commit"):
            if not _auth_required(write=True):
                self._set_headers(HTTPStatus.UNAUTHORIZED)
                self.wfile.write(b"{\n \"error\": \"unauthorized\" \n}")
                self._audit({"ev": "DENY", "method": "POST", "path": "/mem/commit", "reason": "auth_failed"})
                return
            ticket = str(payload.get("ticket", ""))
            ack, conflict = self.store.commit_ticket(ticket)
            if conflict:
                self._set_headers(int(conflict.get("status", 400)))
                self.wfile.write(json.dumps(conflict, ensure_ascii=False).encode("utf-8"))
            else:
                self._set_headers(HTTPStatus.OK)
                self.wfile.write(json.dumps(ack, ensure_ascii=False).encode("utf-8"))
            return

        self._set_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(b"{\n \"error\": \"not found\" \n}")

    def _handle_stream(self):
        # Basic SSE with heartbeats + connection limits
        ip = self.client_address[0]
        try:
            max_total = int(os.environ.get("MEM_MAX_SSE", "64"))
        except Exception:
            max_total = 64
        try:
            max_ip = int(os.environ.get("MEM_MAX_SSE_PER_IP", "16"))
        except Exception:
            max_ip = 16
        if MemHandler.sse_clients >= max_total or MemHandler.sse_by_ip.get(ip, 0) >= max_ip:
            self._set_headers(HTTPStatus.TOO_MANY_REQUESTS)
            self.wfile.write(b"{\n \"error\": \"too_many_sse\" \n}")
            self._audit({"ev": "DENY", "method": "GET", "path": "/mem/stream", "reason": "sse_limit"})
            return
        MemHandler.sse_clients += 1
        MemHandler.sse_by_ip[ip] = MemHandler.sse_by_ip.get(ip, 0) + 1
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", self._cors_origin())
            self.end_headers()

            q = MemHandler.bus.subscribe()
            # initial hello
            self.wfile.write(f"event: hello\ndata: {{\"ts\":\"{utc_ts()}\",\"msg\":\"connected\"}}\n\n".encode("utf-8"))
            self.wfile.flush()
            last_beat = time.time()
            while True:
                # heartbeat every 15s
                if time.time() - last_beat > 15:
                    self.wfile.write(f"event: ping\ndata: {{\"ts\":\"{utc_ts()}\"}}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_beat = time.time()
                try:
                    msg = q.get(timeout=1.0)
                    self.wfile.write(f"event: ev\ndata: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                MemHandler.bus.unsubscribe(q)
            except Exception:
                pass
            # decrement counts
            MemHandler.sse_clients = max(0, MemHandler.sse_clients - 1)
            MemHandler.sse_by_ip[ip] = max(0, MemHandler.sse_by_ip.get(ip, 1) - 1)


def run(host: str = "127.0.0.1", port: int = 8088, store_dir: str = "logus-exaone") -> None:
    # Reuse existing store if already initialized for this directory
    if not isinstance(MemHandler.store, MemStore) or getattr(MemHandler.store, "store_dir", None) != store_dir:
        MemHandler.store = MemStore(store_dir)
    # Start UZR mirror subscriber (항상 구독). 실패 시 필수 모드면 중단.
    if MemHandler.uzr_sync is None:
        try:
            uzr_sync = UzrSync(MemHandler.store, MemHandler.bus)
            # 필수 동작 강제: UZR_REQUIRED=true/1/on 이고 미러 비활성 시 종료
            req = os.environ.get("UZR_REQUIRED", "").strip().lower() in {"1", "true", "on", "yes"}
            if req and not uzr_sync.mirror.enabled:
                raise RuntimeError("UZR mirror required but not enabled (set UZR_CKPT or place default ckpt)")
            uzr_sync.start()
            MemHandler.uzr_sync = uzr_sync
            print("[mem][uzr] mirror thread started (enabled=%s)" % bool(uzr_sync.mirror.enabled))
            # Optional backfill at startup
            try:
                if uzr_sync.mirror.enabled and os.environ.get("UZR_BACKFILL_ON_START", "").strip().lower() in {"1","true","on","yes"}:
                    lim_env = os.environ.get("UZR_BACKFILL_LIMIT", "").strip()
                    lim = int(lim_env) if lim_env.isdigit() else None
                    res = uzr_sync.mirror.backfill(lim)
                    print(f"[mem][uzr] backfill at start: {res}")
            except Exception as e:
                print(f"[mem][uzr] backfill skipped: {e}")
        except Exception as e:
            print(f"[mem][uzr] mirror start failed: {e}")
            if os.environ.get("UZR_REQUIRED", "").strip().lower() in {"1", "true", "on", "yes"}:
                raise
    server = ThreadingHTTPServer((host, port), MemHandler)
    print(f"[mem] listening on http://{host}:{port} (store_dir={store_dir})")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[mem] shutting down...")
    finally:
        server.server_close()


class DreamerConfig:
    def __init__(self, enabled: bool = True, idle_seconds: int = 60, every_seconds: int = 600, window_seconds: int = 4 * 3600):
        self.enabled = enabled
        self.idle_seconds = int(idle_seconds)
        self.every_seconds = int(every_seconds)
        self.window_seconds = int(window_seconds)


class Dreamer(threading.Thread):
    """Background consolidation thread.

    - Waits for idle period, then compacts recent events into Episode nodes and links related items.
    - Uses in-process MemStore write path to preserve L1 hash-chain and L2 consistency.
    """

    def __init__(self, store: MemStore, cfg: DreamerConfig):
        super().__init__(daemon=True)
        self.store = store
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        if not self.cfg.enabled:
            return
        while not self._stop.is_set():
            now = time.time()
            idle_for = now - (self.store.last_event_unix or now)
            if idle_for >= self.cfg.idle_seconds:
                try:
                    self._dream_once()
                except Exception as e:
                    try:
                        MemHandler.bus.publish({"ts": utc_ts(), "ev": "DREAM_ERROR", "error": str(e)})
                    except Exception:
                        pass
                # throttle after a pass
                self._sleep_secs(self.cfg.every_seconds)
            else:
                # poll until idle
                self._sleep_secs(1)

    def _sleep_secs(self, s: int):
        for _ in range(max(1, int(s))):
            if self._stop.is_set():
                return
            time.sleep(1)

    # -------- core logic --------
    def _dream_once(self):
        start_ts = utc_ts()
        MemHandler.bus.publish({"ts": start_ts, "ev": "DREAM_START"})
        t0 = time.time()
        # Read recent events (bounded)
        recent = self._read_recent_events(self.cfg.window_seconds, max_lines=2000)
        # Collect node ids created recently
        node_ids: List[str] = []
        for ev in recent:
            if ev.get("ev") == "WRITE_OK" and ev.get("op") == "UPSERT_NODE" and ev.get("id"):
                node_ids.append(ev["id"])

        # Group by project tag
        groups: Dict[str, List[str]] = {}
        for nid in node_ids:
            n = self.store.nodes.get(nid) or {}
            proj = self._node_project(n) or "_misc"
            groups.setdefault(proj, []).append(nid)

        episodes_created = 0
        edges_added = 0

        for proj, nids in groups.items():
            if len(nids) < 2:
                continue
            # build Episode node
            eid = self._make_episode_id()
            title = f"Episode {start_ts} ({proj})"
            # crude summary: list up to 3 titles
            titles = [self.store.nodes.get(i, {}).get("title", i) for i in nids[:3]]
            body = f"nodes={len(nids)}; top=[" + ", ".join(titles) + "]"
            tags = ["episode", f"proj:{proj}"] if proj != "_misc" else ["episode"]
            ops = [{
                "op": "UPSERT_NODE",
                "type": "Episode",
                "id": eid,
                "title": title,
                "body": body,
                "tags": tags,
                "trust": 0.65,
                "etag": "v1"
            }]
            # edges from episode to members
            for nid in nids[:50]:  # cap fan-out
                ops.append({"op": "ADD_EDGE", "src": eid, "dst": nid, "rel": "relates_to", "weight": 0.8})
            # link discovery among members (limited pairs)
            pairs = self._candidate_pairs(nids[:24])
            for a, b, w in pairs:
                try:
                    ta = set(self.store.nodes.get(a, {}).get("tags", []) or [])
                    tb = set(self.store.nodes.get(b, {}).get("tags", []) or [])
                except Exception:
                    ta, tb = set(), set()
                thr = 0.78
                if ("ern" in ta) and ("ern" in tb):
                    thr = 0.72
                if w >= thr:
                    ops.append({"op": "ADD_EDGE", "src": a, "dst": b, "rel": "relates_to", "weight": round(float(w), 3)})

            ack, conflict = self.store.write({"ops": ops, "author": "dreamer", "justification": f"background consolidation ({proj})"})
            if conflict:
                # continue to next group; conflicts are expected on concurrent edits
                continue
            episodes_created += 1
            # count edges
            for r in ack.get("results", []):
                if isinstance(r, dict) and r.get("edge"):
                    edges_added += 1

        dur = time.time() - t0
        # log stats
        self._append_dreamer_stats({
            "ts": start_ts,
            "duration_s": round(dur, 3),
            "episodes_created": episodes_created,
            "edges_added": edges_added,
        })
        MemHandler.bus.publish({"ts": utc_ts(), "ev": "DREAM_DONE", "episodes": episodes_created, "edges": edges_added, "dur": round(dur, 2)})

    def _append_dreamer_stats(self, row: Dict[str, Any]):
        import csv
        path = os.path.join(self.store.store_dir, "dreamer_stats.csv")
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    def _node_project(self, node: Dict[str, Any]) -> Optional[str]:
        # copy of MemStore._node_project logic without dependency
        for p in node.get("parents", []) or []:
            if isinstance(p, str) and p.startswith("proj:"):
                return p.split(":", 1)[1]
        for t in node.get("tags", []) or []:
            if isinstance(t, str) and t.startswith("proj:"):
                return t.split(":", 1)[1]
        return None

    def _make_episode_id(self) -> str:
        # simple time-based id with randomness
        import random
        return f"ep:{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{random.randint(1000,9999)}"

    def _read_recent_events(self, window_seconds: int, max_lines: int = 2000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        now = time.time()
        path = self.store.l1_path
        if not os.path.exists(path):
            return out
        try:
            # tail-read up to max_lines
            with open(path, "rb") as f:
                # approximate tail by reading last ~256KB
                try:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    read_size = min(262144, size)
                    f.seek(-read_size, os.SEEK_END)
                except Exception:
                    f.seek(0)
                data = f.read().decode("utf-8", errors="ignore")
            lines = [l for l in data.splitlines() if l.strip()][-max_lines:]
            for l in lines:
                try:
                    ev = json.loads(l)
                    ts = ev.get("ts")
                    if ts:
                        import calendar
                        t_struct = time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                        age = now - calendar.timegm(t_struct)
                        if 0 <= age <= window_seconds:
                            out.append(ev)
                except Exception:
                    continue
        except Exception:
            return out
        return out

    def _candidate_pairs(self, nids: List[str]) -> List[Tuple[str, str, float]]:
        # compute pairwise similarities for a small subset
        pairs: List[Tuple[str, str, float]] = []
        vecs: Dict[str, List[float]] = {}
        ids = list(dict.fromkeys(nids))  # unique, keep order
        for nid in ids:
            n = self.store.nodes.get(nid) or {}
            text = f"{n.get('title','')} {n.get('body','')}".strip()
            vecs[nid] = n.get("embedding") or toy_embed(text)
        for i, a in enumerate(ids):
            va = vecs[a]
            for b in ids[i + 1:]:
                vb = vecs[b]
                w = cosine(va, vb)
                pairs.append((a, b, w))
        # sort by similarity desc, return top limited
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs[:64]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="EXAONE×UZR External Memory Gateway (with Dreamer)")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--store_dir", type=str, default="logus-exaone")
    ap.add_argument("--dreamer", type=str, default="on", choices=["on", "off"]) 
    ap.add_argument("--dream_idle", type=int, default=60, help="Idle seconds before a dream pass")
    ap.add_argument("--dream_every", type=int, default=600, help="Min seconds between dream passes")
    ap.add_argument("--dream_window", type=int, default=4 * 3600, help="Recent window (seconds) for consolidation")
    args = ap.parse_args()

    ensure_dir(args.store_dir)

    # start store and dreamer
    MemHandler.store = MemStore(args.store_dir)
    dcfg = DreamerConfig(enabled=(args.dreamer == "on"), idle_seconds=args.dream_idle, every_seconds=args.dream_every, window_seconds=args.dream_window)
    dreamer = Dreamer(MemHandler.store, dcfg)
    dreamer.start()

    # run HTTP server (blocking)
    run(host=args.host, port=args.port, store_dir=args.store_dir)
