# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run sessions for `carveout web`: one RunSession per scene profile.

A session owns the scene's WebRunCore (the shared sequencing core with
web hooks), its event buffer (SSE replay via Last-Event-ID for a
reconnect; a snapshot, the current stage and refusal and a log tail for
a fresh page — the
re-attach contract: the run lives server-side, the browser is a
spectator), and its job slot. GPU stages run in a background thread, ONE
AT A TIME server-wide (_GPU_LOCK); while a job runs, every log record is
captured into the buffer so the browser streams exactly what the terminal
would print. Stage refusals (the core Refusal type) are published
verbatim, never swallowed (driver contract).

Stop-and-look events (factor-cap kills; VIOLATED negative ceiling)
block the job thread until the typed ack round-trips through
/api/runs/<scene>/acks — the run stays halted server-side, surviving any
number of tab closes.
"""

import logging
import queue
import threading
import time
from pathlib import Path

from .. import config as _config
from ..refusal import Refusal, missing_dependency
from ..sequencing import (GATES, Journal, RunCore, StageCancelled,
                          WorkdirLock, WorkdirLocked, fingerprint,
                          )

log = logging.getLogger(__name__)

_GPU_LOCK = threading.Lock()   # one blocking CUDA stage at a time, server-wide


def _release_gpu(kind: str) -> None:
    """gc.collect then torch.cuda.empty_cache, the order LocalBackend.close
    uses: collect first so the cache has something to give back. No-op
    without CUDA; never raises (it runs in a finally)."""
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved()
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved()
            if before - after > 2**28:      # say it only when it mattered
                log.info("%s: released %.1f GiB of GPU workspace "
                         "(%.1f GiB still reserved)", kind,
                         (before - after) / 2**30, after / 2**30)
    except Exception:                       # pragma: no cover - defensive
        log.exception("GPU release after job %s", kind)

# Which gate owns a job's refusal when the raise site did not name one
# (a stage cannot always know: run_detect serves jobs owned by three
# different gates). Without this fallback such refusals surfaced with
# gate=None and the UI could not route the operator back to the panel
# holding the knobs the message names.
JOB_GATE = {"render": "render", "propose_volume": "volume",
            "probe": "vocabulary", "confirm_vocabulary": "vocabulary",
            "propose_prompts": "vocabulary", "threshold_override": "exemplars",
            "reprobe_exemplars": "exemplars", "verify": "verify_consent"}


class EventBuffer:
    """Sequenced event log with live subscribers + replay (Last-Event-ID).

    Two replays: a RECONNECT whose Last-Event-ID is still in
    the buffer gets exactly the events it missed — the re-attach contract
    (the browser reconnects on its own after a blip). A FRESH attach (no
    id, or one the buffer has evicted) gets the journal snapshot the
    server sends first, then only what a page needs to stand where the
    run stands: the latest stage event, the latest refusal while it is
    still current (no attempt started after it), and a tail of the log.
    The 8000-event replay this replaces re-sent every log line and every
    resolved refusal and stop-and-look of the server's life, which the
    reducer then had to un-say."""

    def __init__(self, cap: int = 2000, tail: int = 200):
        self.cap = cap
        self.tail = tail
        self.events: list[tuple[int, str, dict]] = []
        self.seq = 0
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []

    def publish(self, etype: str, payload: dict) -> None:
        with self._lock:
            self.seq += 1
            ev = (self.seq, etype, payload)
            self.events.append(ev)
            if len(self.events) > self.cap:
                del self.events[:len(self.events) - self.cap]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass   # a stalled subscriber loses events; replay recovers

    def subscribe(self, last_id: int = 0):
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._lock:
            oldest = self.events[0][0] if self.events else self.seq + 1
            if last_id and last_id >= oldest - 1:
                replay = [e for e in self.events if e[0] > last_id]
            else:
                replay = self._fresh_replay()
            self._subs.append(q)
        return q, replay

    def _fresh_replay(self) -> list:
        """The trimmed replay for a page that starts from nothing (called
        under the lock): the latest stage event, the latest refusal if no
        stage has started since it, the last `tail` log lines — in
        sequence order, so the browser's Last-Event-ID lands on the newest.
        Acks and stop-and-looks are never replayed: a halted job's open
        blocks ride the journal snapshot instead."""
        latest_stage = next((e for e in reversed(self.events)
                             if e[1] == "stage"), None)
        latest_refusal = next((e for e in reversed(self.events)
                               if e[1] == "refusal"), None)
        picked = []
        if latest_stage:
            picked.append(latest_stage)
        if latest_refusal:
            started_after = any(
                e[1] == "stage" and e[2].get("state") == "running"
                and e[0] > latest_refusal[0] for e in self.events)
            if not started_after:
                picked.append(latest_refusal)
        logs = [e for e in self.events if e[1] == "log"][-self.tail:]
        return sorted(picked + logs, key=lambda e: e[0])

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)


class _JobLogHandler(logging.Handler):
    """Streams every log record emitted while a job runs into the buffer —
    the browser sees exactly the terminal's lines (one job at a time makes
    a global capture faithful)."""

    def __init__(self, publish):
        super().__init__()
        self._publish = publish
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self._publish("log", dict(
                ts=time.strftime("%H:%M:%S"),
                source=record.name.rsplit(".", 1)[-1],
                level=record.levelname, line=self.format(record)))
        except Exception:
            pass


class WebRunCore(RunCore):
    """RunCore hooks answered against the session's event stream/acks."""

    def __init__(self, session, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = session

    def say(self, msg: str) -> None:
        self._session.buffer.publish("log", dict(
            ts=time.strftime("%H:%M:%S"), source="driver", level="INFO",
            line=msg))

    def ack_stop_look(self, kills: list[dict]) -> None:
        # biggest first: the modal's sentence names the first row as the
        # largest, and the typed word is the first row's class
        kills = sorted(kills, key=lambda k: -float(k.get("ratio") or 0))
        payload = dict(
            kills=kills,
            overlays=[f"frame_{k['frame']:04d}.png" for k in kills])
        bid = self._session.open_block("area_factor_kill", payload)
        self._session.buffer.publish("stop_look", dict(
            kind="area_factor_kill", id=bid, **payload))
        # The run halts HERE, server-side, until the typed ack round-trips;
        # a day-long wait beats silently trusting an unreviewed kill.
        if not self._session.wait_block("area_factor_kill", bid,
                                        timeout=24 * 3600):
            raise Refusal("a dropped detection was never looked at; the run "
                          "halted waiting for the acknowledgment and gave up",
                          gate="exemplars")

    def ask_verify_consent(self, n_objects: int, model: str,
                           available: bool) -> bool:
        consent = self._session.pending_consent
        if consent is None:
            raise Refusal("verify run/skip not supplied; answer the "
                          "verify gate first", gate="verify_consent")
        return consent

    def should_cancel(self) -> bool:
        # the session's cancel Event, set by POST /jobs/cancel.
        return self._session._cancel.is_set()


class RunSession:
    """One scene's live server-side state (core, events, job, lock)."""

    def __init__(self, name: str, scene_path: str, workdir: str,
                 cfg_loader):
        self.name = name
        self.buffer = EventBuffer()
        self.core = WebRunCore(self, scene_path, workdir, name, cfg_loader)
        self.wlock = WorkdirLock(workdir, holder="carveout web")
        self._profile_fp = fingerprint(self.core.profile_path)
        self.job: dict | None = None
        self.pending_consent: bool | None = None
        self.vocab_draft: dict | None = None   # {prompts, negatives, probed}
        self._signals: dict[str, threading.Event] = {}
        self._pending: dict[str, str] = {}  # kind -> id of the waiting block
        # kind -> the block's event payload, kept while it waits: the SSE
        # replay buffer is capped, so a long job's logs could evict the
        # stop_look event and a reload then showed a halted run with no
        # dialog and no way to ack it. The journal snapshot re-serves it.
        self._block_payloads: dict[str, dict] = {}
        self._acked: set[str] = set()       # block ids acknowledged
        self._ack_lock = threading.Lock()
        self._cancel = threading.Event()   # set by POST /jobs/cancel
        self._lock = threading.Lock()

    def cancel_job(self) -> bool:
        """Request cancellation of the running stage. The stage polls
        should_cancel() at its next loop boundary and raises StageCancelled;
        idempotent, a no-op when nothing is running."""
        if self.job and self.job["thread"].is_alive() and "outcome" not in self.job:
            self._cancel.set()
            return True
        return False

    # -- signals / acks ---------------------------------------------------------
    # A blocking hook (stop-and-look) registers a block with a
    # fresh id BEFORE publishing the event that carries it, and only an ack
    # naming that id releases it. The id is what makes a stale ack inert: the
    # SSE buffer replays a finished run's stop_look on every page load, and
    # under the old bare-kind scheme re-acking that ghost left the ack armed
    # in a set with no waiter — where the NEXT real never-event consumed it
    # and never blocked at all. A never-event gate must not be satisfiable by
    # an acknowledgment of something else.
    def open_block(self, kind: str, payload: dict | None = None) -> str:
        import uuid
        bid = str(uuid.uuid4())
        with self._ack_lock:
            self._pending[kind] = bid
            self._block_payloads[kind] = payload or {}
            self._signals.setdefault(kind, threading.Event()).clear()
        return bid

    def wait_block(self, kind: str, block_id: str, timeout: float) -> bool:
        """Halt the job thread until the block is acknowledged, the timeout
        passes, or the operator cancels. Cancel used to be unreachable
        here: the wait was on the ack Event alone, so a halted run held the
        card and the job slot for every scene until the word was typed."""
        ev = self._signals[kind]
        with self._ack_lock:
            if block_id in self._acked:   # ack landed before we started waiting
                self._acked.discard(block_id)
                self._pending.pop(kind, None)
                self._block_payloads.pop(kind, None)
                return True
        deadline = time.monotonic() + timeout
        ok = False
        while True:
            if ev.wait(min(1.0, max(0.0, deadline - time.monotonic()))):
                ok = True
                break
            if self._cancel.is_set():
                with self._ack_lock:
                    if self._pending.get(kind) == block_id:
                        self._pending.pop(kind, None)
                        self._block_payloads.pop(kind, None)
                raise StageCancelled("stage cancelled by operator while "
                                     "halted for an acknowledgment")
            if time.monotonic() >= deadline:
                break
        with self._ack_lock:
            self._acked.discard(block_id)
            if self._pending.get(kind) == block_id:
                self._pending.pop(kind, None)
                self._block_payloads.pop(kind, None)
        return ok

    @staticmethod
    def ack_word(v) -> str:
        """An acknowledgment compared loosely: case and whitespace do not
        make a different class name."""
        return " ".join(str(v or "").split()).casefold()

    def expected_ack(self, kind: str) -> str:
        """The word that acknowledges the block currently waiting: the class
        named in the first kill row — typing it is the proof the row was
        read, which is what the typed ack exists for. 'ack' only for
        a block that carries no kill rows."""
        with self._ack_lock:
            kills = (self._block_payloads.get(kind) or {}).get("kills") or []
        return str(kills[0].get("concept") or "ack") if kills else "ack"

    def supply_ack(self, kind: str, block_id: str | None) -> None:
        """POST /acks landed. Refuses anything but the block currently
        waiting — no waiter, or a replayed id, is not an acknowledgment."""
        with self._ack_lock:
            current = self._pending.get(kind)
            if current is None:
                raise Refusal(
                    f"nothing is waiting on a {kind!r} acknowledgment; the "
                    f"run is not halted (a stop that was already answered "
                    f"cannot be answered again)",
                    status=409)
            if block_id != current:
                raise Refusal(
                    f"this {kind!r} acknowledgment is stale; it names a "
                    f"block that is no longer waiting; reload and answer the "
                    f"current one", status=409)
            self._acked.add(block_id)
            ev = self._signals.get(kind)
        if ev is not None:
            ev.set()
        self.buffer.publish("ack", dict(kind=kind, id=block_id))



    # -- workdir lock -------------------------------------------------------------
    def acquire_lock(self) -> None:
        """Take (or re-assert) the single-writer lock; WorkdirLocked
        propagates to a 423 with the holder named."""
        self.wlock.acquire()

    def lock_info(self) -> dict | None:
        rec = self.wlock.read()
        if rec is None:
            return None
        import os
        return dict(holder=rec.get("holder"), pid=rec.get("pid"),
                    since=rec.get("since"),
                    alive=WorkdirLock.alive(rec),
                    mine=int(rec.get("pid") or -1) == os.getpid())

    # -- journal snapshot -----------------------------------------------------------
    def journal_state(self) -> dict:
        """The journal with staleness recomputed from live content hashes —
        run_journal.json semantics ARE the gate-state model; outside edits
        surface here as stale gates."""
        if not (self.job and self.job["thread"].is_alive()):
            # pick up terminal-side / hand edits when we're not driving:
            # re-read the journal, and drop the cached config when the
            # profile changed on disk — a long-lived web session must not
            # run stages on pre-edit values (the terminal reloads per
            # process; the journal's scene_yaml staleness already flags
            # the edit, the VALUES have to follow). Never mid-job: a
            # running stage keeps the config it started with.
            self.core.journal = Journal(self.core.workdir)
            fp = fingerprint(self.core.profile_path)
            if fp != self._profile_fp:
                self._profile_fp = fp
                self.core._invalidate_cfg()
        gates = self.core.gate_states()
        active = next((g for g in GATES
                       if not (gates.get(g)
                               and gates[g]["state"] == "approved")), None)
        with self._ack_lock:
            blocks = [dict(kind=k, id=bid,
                           **self._block_payloads.get(k, {}))
                      for k, bid in self._pending.items()]
        return dict(gates=gates, active_gate=active,
                    running=self.job_state(), lock=self.lock_info(),
                    blocks=blocks,
                    scene=str(self.core.scene),
                    workdir=str(self.core.workdir))

    # -- gate-mutating actions -----------------------------------------------------
    # The ONE place the write invariants live: every endpoint that mutates
    # run state goes through these three operations instead of hand-
    # assembling lock -> mid-job guard -> write -> publish per handler
    # (which is how four separate commits each had to retrofit one
    # invariant onto one handler, and two handlers still lacked the guard).

    def guard_no_job(self, act: str, gate: str | None = None) -> None:
        """Refuse while a job is running: a write under a live job races
        the artifacts the job is reading or rewriting. `act` finishes the
        sentence 'a <kind> job is running — ...'."""
        j = self.job_state()
        if j and j["state"] == "running":
            raise Refusal(f"a {j['kind']} job is running; {act}",
                          gate=gate, status=409)

    def write(self, fn, *, gate: str | None = None,
              during_job: str = "refuse"):
        """A state write under the product's invariants: single-writer
        lock; mid-job guard unless during_job='allow' (reserved for writes
        a job cannot race — the cosmetic viewer settings, the vocab draft
        a job snapshots at start); then fn; then ONE journal snapshot
        published. Returns fn's result."""
        self.acquire_lock()
        if during_job != "allow":
            self.guard_no_job("retry when it finishes", gate=gate)
        result = fn()
        self.buffer.publish("journal", self.journal_state())
        return result

    def approve_gate(self, gate: str, precheck=None, **extra) -> dict:
        """The approve act under the invariants. An approval pins content
        hashes; approving while a job rewrites the very artifacts being
        pinned records a hash of a moving target — hence the guard. The
        gate's own precheck (domain checks) runs after it, so a mid-job
        refusal keeps its clearer message. Returns the fresh snapshot."""
        self.acquire_lock()
        self.guard_no_job("approve when it finishes, so the approval "
                          "covers what is actually on disk",
                          gate=gate if gate in GATES else None)
        if precheck is not None:
            precheck()
        self.core.approve(gate, **extra)
        self.buffer.publish("journal", self.journal_state())
        return self.journal_state()

    def reopen_gate(self, gate: str) -> tuple[list[str], dict]:
        """The reopen act, same mid-job guard as approve: demoting journal
        records a running job is about to rewrite invites the two to
        interleave. Returns (demoted gates, fresh snapshot)."""
        self.acquire_lock()
        self.guard_no_job("reopen when it finishes", gate=gate)
        demoted = self.core.reopen(gate)
        self.buffer.publish("journal", self.journal_state())
        return demoted, self.journal_state()

    def job_state(self) -> dict | None:
        j = self.job
        if not j:
            return None
        # A job is running only while its thread is alive AND it has not
        # recorded an outcome: the end-of-job journal event is published
        # FROM the job thread (still alive at that instant), and a bare
        # is_alive() made that final event claim state=running — every
        # panel stayed disabled until a reload (seen on the first
        # live gate walk).
        running = j["thread"].is_alive() and "outcome" not in j
        return dict(kind=j["kind"], started=j["started"],
                    state="running" if running else j.get("outcome",
                                                          "done"))

    # -- jobs -----------------------------------------------------------------------
    def ensure_job_slot(self) -> None:
        """Refuse NOW — before the caller mutates anything — if start_job
        would refuse. Routes that write state and then start a job call this
        first, so the refusal lands before the write instead of after it (a
        reset that deletes stage1 and THEN learns a render is running has
        destroyed the very frames that render was writing). Advisory, not a
        reservation: start_job still re-checks atomically."""
        with self._lock:
            if self.job and self.job["thread"].is_alive():
                raise Refusal(f"a job is already running for this scene: "
                              f"{self.job['kind']}", status=409)
            if not _GPU_LOCK.acquire(blocking=False):
                raise Refusal("another stage is already running (one GPU "
                              "job at a time)", status=409)
            _GPU_LOCK.release()
            self.acquire_lock()

    def job_gate(self) -> str | None:
        """The gate that owns the RUNNING job's refusals: the explicit gate
        its starter named (a calibration job's affected gate, which JOB_GATE
        cannot know — the same kind serves every gate), else the kind's
        static mapping."""
        j = self.job
        return j and (j.get("gate") or JOB_GATE.get(j["kind"]))

    def start_job(self, kind: str, fn, gate: str | None = None) -> None:
        """Run fn in a background thread; one job per server (GPU rule).
        Every exit path publishes a stage event; refusals go out verbatim,
        routed to `gate` when given (else JOB_GATE by kind)."""
        with self._lock:
            if self.job and self.job["thread"].is_alive():
                raise Refusal(f"a job is already running for this scene: "
                              f"{self.job['kind']}", status=409)
            if not _GPU_LOCK.acquire(blocking=False):
                raise Refusal("another stage is already running (one GPU "
                              "job at a time)", status=409)
            try:
                self.acquire_lock()
            except WorkdirLocked:
                _GPU_LOCK.release()
                raise
            self._cancel.clear()   # fresh cancel state per job
            thread = threading.Thread(target=self._run_job, args=(kind, fn),
                                      daemon=True, name=f"job-{self.name}")
            self.job = dict(kind=kind, thread=thread, gate=gate,
                            started=time.strftime("%Y-%m-%d %H:%M:%S"))
            thread.start()

    def _run_job(self, kind: str, fn) -> None:
        handler = _JobLogHandler(self.buffer.publish)
        logging.getLogger().addHandler(handler)
        self.buffer.publish("stage", dict(stage=kind, state="running",
                                          started=self.job["started"]))
        try:
            fn()
            self.job["outcome"] = "done"
            self.buffer.publish("stage", dict(stage=kind, state="done"))
        except StageCancelled:
            # operator stop. No manifest was written (manifest-last), so
            # the stage stays re-runnable; publish a distinct terminal state.
            self.job["outcome"] = "cancelled"
            self.buffer.publish("stage", dict(stage=kind, state="cancelled"))
        except Refusal as e:
            # The core refusal type — surfaced verbatim, never swallowed
            # (driver contract), routed to the gate the raise site named
            # (else the running job's owning gate).
            self.job["outcome"] = "failed"
            self.buffer.publish("refusal", dict(
                message=str(e), gate=e.gate or self.job_gate(),
                remedy=e.remedy))
            self.buffer.publish("stage", dict(stage=kind, state="failed"))
        except SystemExit as e:
            # Fallback only: stages raise Refusal now; a SystemExit from
            # third-party code still surfaces verbatim rather than dying.
            self.job["outcome"] = "failed"
            msg = e.code if isinstance(e.code, str) else f"exit {e.code}"
            self.buffer.publish("refusal", dict(message=msg,
                                                gate=self.job_gate(),
                                                remedy=None))
            self.buffer.publish("stage", dict(stage=kind, state="failed"))
        except ModuleNotFoundError as e:
            # torch, gsplat or sam3 absent: the server starts without them
            # (nothing on the request path imports a CUDA stack), so the
            # first stage is where a bad environment shows — as the remedy,
            # not a traceback
            self.job["outcome"] = "failed"
            r = missing_dependency(e.name or str(e), e)
            self.buffer.publish("refusal", dict(
                message=str(r), gate=self.job_gate(), remedy=None))
            self.buffer.publish("stage", dict(stage=kind, state="failed"))
        except Exception as e:
            self.job["outcome"] = "failed"
            log.exception("job %s failed", kind)
            self.buffer.publish("refusal", dict(
                message=f"{type(e).__name__}: {e}", gate=self.job_gate(),
                remedy=None))
            self.buffer.publish("stage", dict(stage=kind, state="failed"))
        finally:
            logging.getLogger().removeHandler(handler)
            # Give the card back at EVERY job boundary, success or failure:
            # a failed job's traceback keeps its frame locals (the splat
            # tensors, a half-built model) alive until a collection, and
            # PyTorch keeps freed blocks reserved until asked — reserved
            # memory the next stage's preflight cannot see as free.
            _release_gpu(kind)
            _GPU_LOCK.release()
            try:
                self.buffer.publish("journal", self.journal_state())
            except Exception:
                log.exception("journal snapshot after job %s", kind)


class RunManager:
    """Scene registry (profiles with scene_path+workdir, merged across the
    profile search path, are the library) + session cache; releases every
    held lock on shutdown."""

    def __init__(self, repo: Path, cfg_loader_factory):
        self.repo = repo
        self.cfg_loader_factory = cfg_loader_factory   # name -> cfg dict
        self.sessions: dict[str, RunSession] = {}
        self._lock = threading.Lock()

    def profiles(self) -> dict[str, dict]:
        """name -> dict(raw=profile): every profile with scene_path +
        workdir in the one profile directory (read at call time, so a
        `--profiles-dir` set before serving is honoured)."""
        import yaml
        out = {}
        for p in sorted(_config.PROFILES_DIR.glob("*.yaml")):
            try:
                raw = yaml.safe_load(p.read_text()) or {}
            except Exception as e:
                # A profile that does not parse stays IN the library, as a
                # row that says so — it used to vanish without a word.
                out[p.stem] = dict(raw=None, error=(
                    f"{p} does not parse as YAML ({type(e).__name__}: "
                    f"{e}); fix the file, or restore it from its "
                    f".orig copy if one exists beside it"))
                log.warning("profile %s unreadable: %s", p, e)
                continue
            if not isinstance(raw, dict):
                out[p.stem] = dict(raw=None, error=(
                    f"{p} is not a profile (a YAML mapping); fix the "
                    f"file or remove it"))
                continue
            if raw.get("scene_path") and raw.get("workdir"):
                out[p.stem] = dict(raw=raw)
        return dict(sorted(out.items()))

    def session(self, name: str) -> RunSession:
        with self._lock:
            if name in self.sessions:
                return self.sessions[name]
            info = self.profiles().get(name)
            if info is None:
                raise Refusal(f"unknown scene {name!r} (no profile with "
                              f"scene_path+workdir in "
                              f"{_config.PROFILES_DIR})", status=404)
            if info.get("error"):
                raise Refusal(info["error"], status=500)
            raw = info["raw"]
            # scene_path/workdir stay relative to the repo root regardless
            # of where the profile file lives
            s = RunSession(name, str(self.repo / raw["scene_path"]),
                           str(self.repo / raw["workdir"]),
                           self.cfg_loader_factory)
            self.sessions[name] = s
            return s

    def forget(self, name: str) -> None:
        """Drop a cached session — the scene is gone from disk, so anything
        still holding its paths would answer about a run that no longer
        exists."""
        with self._lock:
            self.sessions.pop(name, None)

    def stop_all(self, timeout: float = 15.0) -> None:
        """Shutdown: ask every running job to stop and wait for it, so a
        stage is not killed mid-write. Job threads are daemons — a stage
        that ignores the cancel probe cannot hold the exit hostage — but
        every stage checks it at its loop boundaries and every state file
        is written atomically, so the wait normally ends with the manifest
        either fully written or not written at all."""
        for s in list(self.sessions.values()):
            j = s.job
            if j and j["thread"].is_alive():
                s._cancel.set()
        deadline = time.monotonic() + timeout
        for s in list(self.sessions.values()):
            j = s.job
            if j and j["thread"].is_alive():
                j["thread"].join(max(0.0, deadline - time.monotonic()))
                if j["thread"].is_alive():
                    log.warning("job %s for %r still running at shutdown "
                                "(its manifest, if any, is not written)",
                                j["kind"], s.name)

    def release_all(self) -> None:
        for s in self.sessions.values():
            try:
                s.wlock.release()
            except Exception:
                pass
