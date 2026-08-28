"""Record continuous motion and replay it, both arms on one timeline.

The pose-based sequencer captured a handful of static configurations and drove
trapezoid moves between them. That cannot express a hand-drawn path: the shape
of the motion between poses is invented by the profile generator, not recorded.

This records the *whole movement* - every joint, sampled on a clock - while the
operator drags the IK targets. Both arms are written into the same sample, so
they are simultaneous by construction rather than by being started together and
hoping they stay in step.

**Joint angles are recorded, not TCP positions.** Replaying joint angles
reproduces exactly what happened. Replaying Cartesian targets would re-run IK,
and the solver can pick a different arm configuration for the same point -
elbow up instead of elbow down - so the replay would not match the take.
"""

from __future__ import annotations

import json
import math
import time

#: Samples per second. 20 Hz is smooth for arm motion and keeps files small;
#: the sim runs far faster but nothing about a hand-dragged path needs that.
RECORD_HZ = 20.0

#: Largest joint step allowed between consecutive replayed samples, radians.
#: A recording made by hand is inherently rate-limited, but a speed multiplier
#: above 1 compresses time and can outrun what the motor should do, so the
#: player enforces a ceiling of its own.
MAX_REPLAY_STEP_RAD = math.radians(6.0)


class MotionRecorder:
    """Captures ``{joint: angle}`` samples against a monotonic clock."""

    def __init__(self, rate_hz: float = RECORD_HZ):
        self.period = 1.0 / max(rate_hz, 1.0)
        self.samples: list[dict] = []
        self._recording = False
        self._t0 = 0.0
        self._last = 0.0

    # -- recording ---------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        self.samples = []
        self._recording = True
        self._t0 = time.monotonic()
        self._last = -1e9                     # force a sample on the first tick

    def stop(self) -> None:
        self._recording = False

    def capture(self, angles: dict) -> bool:
        """Take one sample if the clock says it is due. Returns True if taken."""
        if not self._recording:
            return False
        now = time.monotonic()
        if now - self._last < self.period:
            return False
        self._last = now
        self.samples.append({"t": round(now - self._t0, 4),
                             "q": {k: round(float(v), 5) for k, v in angles.items()}})
        return True

    # -- playback ----------------------------------------------------------------

    @property
    def duration(self) -> float:
        return self.samples[-1]["t"] if self.samples else 0.0

    def at(self, t: float) -> dict:
        """Joint angles at time ``t``, linearly interpolated between samples.

        Interpolated rather than snapped to the nearest sample so playback is
        smooth at any speed, and so a 20 Hz recording does not become 20 Hz
        stair-stepping on a motor.
        """
        if not self.samples:
            return {}
        if t <= self.samples[0]["t"]:
            return dict(self.samples[0]["q"])
        if t >= self.samples[-1]["t"]:
            return dict(self.samples[-1]["q"])
        lo, hi = 0, len(self.samples) - 1
        while hi - lo > 1:                     # binary search the bracketing pair
            mid = (lo + hi) // 2
            if self.samples[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[hi]
        span = b["t"] - a["t"]
        frac = 0.0 if span <= 0 else (t - a["t"]) / span
        out = {}
        for name, va in a["q"].items():
            vb = b["q"].get(name, va)
            out[name] = va + (vb - va) * frac
        return out

    # -- files -------------------------------------------------------------------

    def save(self, path: str) -> int:
        with open(path, "w") as fh:
            json.dump({"rate_hz": 1.0 / self.period, "samples": self.samples},
                      fh, indent=1)
        return len(self.samples)

    def load(self, path: str) -> int:
        with open(path) as fh:
            data = json.load(fh)
        self.samples = list(data.get("samples", []))
        return len(self.samples)


class MultiPlayer:
    """Plays several tracks at once off ONE clock.

    Each arm is recorded on its own, because a mouse can only drag one target
    at a time. Replay merges the takes: every track is started from t=0 against
    a single clock, so the arms move together even though they were never
    performed together.

    Tracks must own disjoint joints - the dashboard records only the joints of
    each arm's IK chain, and the End effector tab already warns when two chains
    overlap. If they did overlap, whichever track was merged last would win and
    the other arm's take would be silently partly ignored.

    Tracks of different length are looped over the LONGEST duration, with a
    shorter one holding its final pose until the cycle restarts. Looping each on
    its own period would let the arms drift out of phase a little more on every
    repeat, which is rarely what anyone wants from a two-arm routine.
    """

    def __init__(self, tracks: "list[MotionRecorder]"):
        self.tracks = tracks
        self.playing = False
        self.loop = False
        self.speed = 1.0
        self._t0 = 0.0
        self._previous: dict = {}

    @property
    def duration(self) -> float:
        return max((t.duration for t in self.tracks), default=0.0)

    def start(self) -> None:
        if self.duration <= 0:
            return
        self.playing = True
        self._t0 = time.monotonic()
        self._previous = {}

    def stop(self) -> None:
        self.playing = False

    def step(self) -> dict:
        """Merged angles for right now, rate-limited. Empty when not playing."""
        if not self.playing or self.duration <= 0:
            return {}
        elapsed = (time.monotonic() - self._t0) * max(self.speed, 0.01)
        if elapsed > self.duration:
            if not self.loop:
                self.playing = False
                merged = {}
                for track in self.tracks:
                    if track.samples:
                        merged.update(track.samples[-1]["q"])
                return merged
            self._t0 = time.monotonic()
            elapsed = 0.0

        merged = {}
        for track in self.tracks:
            if track.samples:
                merged.update(track.at(elapsed))

        limited = {}
        for name, value in merged.items():
            previous = self._previous.get(name)
            if previous is None:
                limited[name] = value
            else:
                delta = value - previous
                if abs(delta) > MAX_REPLAY_STEP_RAD:
                    delta = math.copysign(MAX_REPLAY_STEP_RAD, delta)
                limited[name] = previous + delta
        self._previous = limited
        return limited

    def progress(self) -> float:
        if not self.playing or self.duration <= 0:
            return 0.0
        elapsed = (time.monotonic() - self._t0) * max(self.speed, 0.01)
        return min(elapsed / self.duration, 1.0)


class MotionPlayer:
    """Walks a single recording against wall-clock time."""

    def __init__(self, recorder: MotionRecorder):
        self.rec = recorder
        self.playing = False
        self.loop = False
        self.speed = 1.0
        self._t0 = 0.0
        self._previous: dict = {}

    def start(self) -> None:
        if not self.rec.samples:
            return
        self.playing = True
        self._t0 = time.monotonic()
        self._previous = {}

    def stop(self) -> None:
        self.playing = False

    def step(self) -> dict:
        """Angles for right now, rate-limited. Empty dict when not playing."""
        if not self.playing or not self.rec.samples:
            return {}
        elapsed = (time.monotonic() - self._t0) * max(self.speed, 0.01)
        if elapsed > self.rec.duration:
            if not self.loop:
                self.playing = False
                return dict(self.rec.samples[-1]["q"])
            self._t0 = time.monotonic()
            elapsed = 0.0
        target = self.rec.at(elapsed)

        # Ceiling on how far any joint may move between frames. The recording is
        # already smooth at 1x, but a speed multiplier compresses time and could
        # otherwise hand the hardware a step far larger than it was recorded at.
        limited = {}
        for name, value in target.items():
            previous = self._previous.get(name)
            if previous is None:
                limited[name] = value
            else:
                delta = value - previous
                if abs(delta) > MAX_REPLAY_STEP_RAD:
                    delta = math.copysign(MAX_REPLAY_STEP_RAD, delta)
                limited[name] = previous + delta
        self._previous = limited
        return limited

    def progress(self) -> float:
        if not self.playing or self.rec.duration <= 0:
            return 0.0
        elapsed = (time.monotonic() - self._t0) * max(self.speed, 0.01)
        return min(elapsed / self.rec.duration, 1.0)