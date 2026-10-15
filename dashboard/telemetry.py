#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# telemetry.py
#
# JSONL telemetry transport between the controller/inference loop and the
# Dash monitoring dashboard. Standard library only, so it can be imported
# from both the experiment venv (writer side) and the dashboard venv (reader
# side).
#
# Writer: the control loop calls emit(record), which only does a
# non-blocking queue.put_nowait(). A daemon thread serialises records and
# appends them to the file, flushing every `flush_every` records or every
# `flush_interval_s` seconds, whichever comes first. If the queue is ever
# full the record is dropped (and counted) rather than blocking the loop.
#
# Reader: incrementally tails the file from the last consumed byte offset,
# only parses complete lines (a trailing partial line is left for the next
# poll), skips malformed lines, and detects file rotation/truncation (a new
# run) by inode change or the file shrinking.
#
# Record types ("type" field):
#   run_start      - run metadata (algo, stage, model paths, action labels)
#   episode_reset  - emitted just before env reset (takeoff can take seconds)
#   step           - one environment step
#   episode_end    - per-episode summary (same metric definitions as 07)
#   run_end        - run finished (with measured step rate)
# ---------------------------------------------------------------------------
import collections
import json
import math
import os
import queue
import threading
import time

DEFAULT_TELEMETRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "runtime", "telemetry.jsonl"
)


def _json_default(o):
    # numpy scalars / arrays without importing numpy here
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _sanitize(v):
    # Strict JSON has no NaN/Inf - publish them as null.
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


_STOP = object()  # writer-thread shutdown sentinel
STEP_REQUIRED_KEYS = ("episode", "episode_step", "global_step")


class TelemetryWriter:
    def __init__(self, path=DEFAULT_TELEMETRY_PATH, flush_every=10,
                 flush_interval_s=0.25, max_queue=20_000, archive_previous=True):
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self.flush_interval_s = flush_interval_s
        self.dropped = 0
        self.written = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if archive_previous and os.path.exists(path) and os.path.getsize(path) > 0:
            # Keep previous runs for inspection; the dashboard always tails a
            # fresh file per run.
            archive_dir = os.path.join(os.path.dirname(path), "archive")
            os.makedirs(archive_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(path)))
            base = os.path.splitext(os.path.basename(path))[0]
            os.replace(path, os.path.join(archive_dir, f"{base}_{stamp}.jsonl"))
        self._queue = queue.Queue(maxsize=max_queue)
        self._fh = open(path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, name="telemetry-writer", daemon=True)
        self._thread.start()

    def emit(self, record):
        """Non-blocking. Never raises into the control loop."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _write(self, rec):
        try:
            clean = {k: _sanitize(v) for k, v in rec.items()}
            self._fh.write(json.dumps(clean, default=_json_default, separators=(",", ":")) + "\n")
            self.written += 1
            return True
        except Exception:
            self.dropped += 1
            return False

    def _run(self):
        pending = 0
        last_flush = time.monotonic()
        while True:
            try:
                rec = self._queue.get(timeout=self.flush_interval_s)
            except queue.Empty:
                rec = None
            if rec is _STOP:
                break
            if rec is not None and self._write(rec):
                pending += 1
            now = time.monotonic()
            if pending and (pending >= self.flush_every or now - last_flush >= self.flush_interval_s):
                self._fh.flush()
                pending = 0
                last_flush = now
        # Drain whatever is left, then flush and close.
        while True:
            try:
                rec = self._queue.get_nowait()
            except queue.Empty:
                break
            if rec is not _STOP:
                self._write(rec)
        self._fh.flush()
        self._fh.close()

    def close(self, timeout=5.0):
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)



class TelemetryReader:
    """Incremental, crash-tolerant JSONL tail used by the dashboard."""

    # Skip straight to the tail of very large files on first open; the
    # run_start line is still read from the head of the file.
    MAX_INITIAL_READ_BYTES = 8 * 1024 * 1024

    def __init__(self, path=DEFAULT_TELEMETRY_PATH, history=600, max_episodes=50):
        self.path = path
        self.history = history
        self.max_episodes = max_episodes
        self._lock = threading.Lock()
        self._reset_state()

    def _reset_state(self):
        self._inode = None
        self._offset = 0
        self._skip_partial = False
        self.meta = None
        self.steps = collections.deque(maxlen=self.history)
        self.episodes = collections.deque(maxlen=self.max_episodes)
        self.last_record = None
        self.last_step = None
        self.run_end = None
        self.bad_lines = 0
        self.last_wall_time = None  # wall time of the newest record

    def poll(self):
        with self._lock:
            try:
                st = os.stat(self.path)
            except FileNotFoundError:
                if self._inode is not None:
                    self._reset_state()
                return
            if self._inode is not None and (st.st_ino != self._inode or st.st_size < self._offset):
                self._reset_state()  # rotated / truncated -> new run
            if self._inode is None:
                self._inode = st.st_ino
                if st.st_size > self.MAX_INITIAL_READ_BYTES:
                    self._read_head_meta()
                    self._offset = st.st_size - self.MAX_INITIAL_READ_BYTES
                    self._skip_partial = True
                else:
                    self._skip_partial = False
            if st.st_size == self._offset:
                return
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read(st.st_size - self._offset)
            last_nl = chunk.rfind(b"\n")
            if last_nl < 0:
                return  # only a partial line so far
            complete = chunk[: last_nl + 1]
            self._offset += len(complete)
            lines = complete.split(b"\n")
            if self._skip_partial:
                lines = lines[1:]
                self._skip_partial = False
            for raw in lines:
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    self.bad_lines += 1
                    continue
                if isinstance(rec, dict):
                    self._ingest(rec)

    def _read_head_meta(self):
        try:
            with open(self.path, "rb") as fh:
                rec = json.loads(fh.readline())
            if rec.get("type") == "run_start":
                self.meta = rec
        except Exception:
            pass

    def _ingest(self, rec):
        t = rec.get("type")
        if t == "step" and not all(isinstance(rec.get(k), int) for k in STEP_REQUIRED_KEYS):
            self.bad_lines += 1  # valid JSON but not a complete step record
            return
        self.last_record = rec
        if isinstance(rec.get("timestamp"), (int, float)):
            self.last_wall_time = rec["timestamp"]
        if t == "run_start":
            self.meta = rec
        elif t == "step":
            self.steps.append(rec)
            self.last_step = rec
        elif t == "episode_end":
            self.episodes.append(rec)
        elif t == "run_end":
            self.run_end = rec

    def snapshot(self):
        with self._lock:
            return dict(
                meta=self.meta,
                steps=list(self.steps),
                episodes=list(self.episodes),
                last_record=self.last_record,
                last_step=self.last_step,
                run_end=self.run_end,
                last_wall_time=self.last_wall_time,
                bad_lines=self.bad_lines,
                file_exists=self._inode is not None,
                version=f"{self._inode}:{self._offset}",  # changes only when new data is consumed
            )
