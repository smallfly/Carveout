# SPDX-FileCopyrightText: 2026 Dpt. <https://www.dpt.co>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seconds-fast unit test for `carveout.sequencing` (pure logic: a temporary
profile directory and workdir, no scene, no models, no GPU): the journal
and its content-hash cascade — approve, reopen, a changed input, a
calibration write that re-opens its gate before the file changes, the
calibration whitelist, a torn journal that refuses by name, atomic writes,
and the single-writer lock.

    python -m tests.test_sequencing
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from carveout.config import load_config, set_profiles_dir
from carveout.refusal import Refusal
from carveout.sequencing import (GATES, Journal, RunCore, WorkdirLock,
                                 WorkdirLocked, write_text_atomic)


class _Core(RunCore):
    """RunCore with the operator lines collected instead of printed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.said: list[str] = []

    def say(self, msg: str) -> None:
        self.said.append(msg)


def _scene(tmp: Path, name: str = "unit") -> _Core:
    """A scaffolded scene in `tmp`: profile in tmp/profiles, workdir in
    tmp/work. The splat file need not exist — nothing here reads it."""
    (tmp / "profiles").mkdir()
    set_profiles_dir(tmp / "profiles")
    core = _Core(str(tmp / "scene" / "scene.ply"), str(tmp / "work"), name,
                 lambda path: load_config(None, scene_config=path))
    core.scaffold({"scene": {"scale_m_per_unit": 1.0},
                   "render": {"path_mode": "interior"}})
    return core


def _states(core: RunCore) -> dict:
    return {g: (s and s["state"]) for g, s in core.gate_states().items()}


def _refuses(fn, *needles, status=None):
    try:
        fn()
    except Refusal as e:
        for n in needles:
            assert n in str(e), (n, str(e))
        if status is not None:
            assert e.status == status, e.status
    else:
        raise AssertionError("expected a Refusal")


def test_cascade():
    tmp = Path(tempfile.mkdtemp())
    try:
        core = _scene(tmp)
        (core.workdir / "volume.json").write_text('{"boxes": []}')
        for g in GATES:
            core.approve(g)
        assert all(s == "approved" for s in _states(core).values())
        # a changed input demotes from the earliest gate that reads it:
        # every gate pins volume.json, so all five read stale
        (core.workdir / "volume.json").write_text('{"boxes": [1]}')
        st = core.gate_states()
        assert all(s["state"] == "stale" for s in st.values())
        assert st["volume"]["changed"] == ["volume"]
        # approving again pins the new content and clears every later gate
        core.approve("volume")
        assert _states(core) == dict(volume="approved", render=None,
                                     vocabulary=None, exemplars=None,
                                     verify_consent=None)
        for g in GATES[1:]:
            core.approve(g)
        # reopen: the gate and its successors lose their approval
        assert core.reopen("vocabulary") == ["vocabulary", "exemplars",
                                             "verify_consent"]
        assert _states(core) == dict(volume="approved", render="approved",
                                     vocabulary=None, exemplars=None,
                                     verify_consent=None)
        # the journal on disk says the same
        again = RunCore(str(core.scene), str(core.workdir), core.name,
                        core.cfg_loader)
        assert _states(again) == _states(core)
    finally:
        shutil.rmtree(tmp)


def test_calibration_write():
    tmp = Path(tempfile.mkdtemp())
    try:
        core = _scene(tmp)
        for g in GATES:
            core.approve(g)
        # the whitelist names the earliest gate a key re-opens
        assert core.calibration_gate({"render": {"num_views": 5}}) == "render"
        assert core.calibration_gate(
            {"detect": {"exemplar_threshold_overrides": {"cup": 0.3}}}) == "exemplars"
        assert core.calibration_gate({"scene": {"floor": 0.1}}) == "volume"
        _refuses(lambda: core.calibration_gate({"detect": {"prompts": ["x"]}}),
                 "not an adjustable calibration key", status=400)
        # a render-calibration write: the profile changes, the render gate
        # and everything after re-open, the volume gate keeps its approval
        # over the new file (restamped, not stale)
        core.write_calibration({"render": {"num_views": 5}}, "render", "test")
        assert core.cfg()["render"]["num_views"] == 5
        assert _states(core) == dict(volume="approved", render=None,
                                     vocabulary=None, exemplars=None,
                                     verify_consent=None)
        # an exemplar-threshold write replaces the record whole and
        # re-opens the exemplars gate only
        for g in GATES[1:]:
            core.approve(g)
        core.write_calibration(
            {"detect": {"exemplar_threshold_overrides": {"cup": 0.3}}},
            "exemplars", "test")
        assert core.cfg()["detect"]["exemplar_threshold_overrides"] == {"cup": 0.3}
        core.approve("exemplars")
        core.write_calibration(
            {"detect": {"exemplar_threshold_overrides": {"mug": 0.2}}},
            "exemplars", "test")
        assert core.cfg()["detect"]["exemplar_threshold_overrides"] == {"mug": 0.2}
        assert _states(core) == dict(volume="approved", render="approved",
                                     vocabulary="approved", exemplars=None,
                                     verify_consent=None)
    finally:
        shutil.rmtree(tmp)


def test_journal_file():
    tmp = Path(tempfile.mkdtemp())
    try:
        # atomic: the file is whole or absent, and no temp file survives
        write_text_atomic(tmp / "a" / "b.json", "{}")
        assert (tmp / "a" / "b.json").read_text() == "{}"
        assert [p.name for p in (tmp / "a").iterdir()] == ["b.json"]
        # a torn journal refuses by name, with the remedy, never a traceback
        (tmp / "run_journal.json").write_text('{"gates": {"vol')
        _refuses(lambda: Journal(tmp), "run_journal.json", "not valid JSON",
                 "delete it", status=500)
        (tmp / "run_journal.json").write_text('[1, 2]')
        _refuses(lambda: Journal(tmp), "not a gate record", status=500)
        os.unlink(tmp / "run_journal.json")
        j = Journal(tmp)
        j.record("volume", {"x": "1"})
        assert json.loads((tmp / "run_journal.json").read_text())["gates"]["volume"]["inputs"] == {"x": "1"}
    finally:
        shutil.rmtree(tmp)


def test_lock():
    tmp = Path(tempfile.mkdtemp())
    try:
        mine = WorkdirLock(tmp, holder="test")
        mine.acquire()
        mine.acquire()                       # re-entrant for the same process
        rec = mine.read()
        assert rec["pid"] == os.getpid() and rec["holder"] == "test"
        # another holder: a live pid refuses, a dead one reads stale and is
        # released only by the explicit act
        other = WorkdirLock(tmp, holder="other")
        (tmp / "run_lock.json").write_text(json.dumps(
            dict(holder="gone", pid=2 ** 22 + 12345, since="never")))
        try:
            other.acquire()
        except WorkdirLocked as e:
            assert e.stale
        else:
            raise AssertionError("expected WorkdirLocked")
        other.break_stale()
        other.acquire()
        other.release()
        assert not (tmp / "run_lock.json").exists()
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print(f"all passed in {time.perf_counter() - t0:.2f} s")
