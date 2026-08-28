"""One self-contained IK control panel: end effector, TCP, target, Follow.

Factored out of the dashboard so the same controls can be instantiated once per
arm. Each panel owns its own chain, its own draggable target and its own solver
state, so two arms are solved independently in the same frame.

The one thing panels must NOT do is share joints. Two solvers that both command
the same joint will fight over it every frame - each undoing the other's
correction - and the arm oscillates instead of converging. ``warn_if_shared``
detects that and says so rather than letting it look like a tuning problem.
"""

from __future__ import annotations

import math
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from ik import ArmChain, candidate_bodies, estimate_tcp

#: IK iterations per frame, per panel. Small because the solution is carried
#: between frames - the solver converges over several frames while the servos
#: are still moving, rather than restarting from scratch each time.
ITERS_PER_FRAME = 4

#: Default cap on commanded joint speed, deg/s. The solver converges far faster
#: than a motor should move: unclamped, a single frame moved a joint 22.9 deg,
#: which at ~1900 solves/s is a 43,000 deg/s setpoint going to the hardware.
DEFAULT_MAX_SPEED_DPS = 30.0


class IKPanel:
    """IK controls for one end effector, drawn into ``parent``."""

    def __init__(self, parent, model, data, label: str, mocap_id: int | None,
                 default_body: str | None = None):
        self.model = model
        self.data = data
        self.label = label
        self.mocap_id = mocap_id
        self.chain: ArmChain | None = None
        self.q_ik: np.ndarray | None = None
        self._last_t = time.monotonic()

        box = ttk.LabelFrame(parent, text=f"Arm {label}")
        box.pack(side="top", fill="x", pady=4, padx=2)

        row = ttk.Frame(box)
        row.pack(side="top", fill="x", pady=2, padx=4)
        self.active = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Follow", variable=self.active).pack(side="left")
        ttk.Label(row, text="End effector").pack(side="left", padx=(10, 2))
        self.body_var = tk.StringVar()
        choices = candidate_bodies(model) or ["world"]
        self.body_var.set(default_body if default_body in choices else choices[0])
        combo = ttk.Combobox(row, textvariable=self.body_var, values=choices,
                             state="readonly", width=24)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.rebuild())

        self.chain_lbl = ttk.Label(box, wraplength=380, justify="left",
                                   foreground="#555")
        self.chain_lbl.pack(side="top", anchor="w", padx=4)

        tcp = ttk.Frame(box)
        tcp.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(tcp, text="TCP offset").pack(side="left")
        self.tcp_vars = []
        for axis in ("X", "Y", "Z"):
            ttk.Label(tcp, text=axis).pack(side="left", padx=(6, 1))
            v = tk.DoubleVar(value=0.0)
            ttk.Entry(tcp, width=6, textvariable=v).pack(side="left")
            self.tcp_vars.append(v)
        ttk.Button(tcp, text="Apply", width=6,
                   command=self.apply_tcp).pack(side="left", padx=(6, 2))
        ttk.Button(tcp, text="Estimate", width=8,
                   command=self.estimate).pack(side="left")

        tgt = ttk.Frame(box)
        tgt.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(tgt, text="Target").pack(side="left")
        self.tgt_vars = []
        for axis in ("X", "Y", "Z"):
            ttk.Label(tgt, text=axis).pack(side="left", padx=(6, 1))
            v = tk.DoubleVar(value=0.0)
            ttk.Entry(tgt, width=6, textvariable=v).pack(side="left")
            self.tgt_vars.append(v)
        ttk.Button(tgt, text="Snap to TCP", width=11,
                   command=self.snap).pack(side="left", padx=6)

        opt = ttk.Frame(box)
        opt.pack(side="top", fill="x", padx=4, pady=2)
        ttk.Label(opt, text="Max speed (deg/s)").pack(side="left")
        self.speed = ttk.Entry(opt, width=6)
        self.speed.insert(0, str(int(DEFAULT_MAX_SPEED_DPS)))
        self.speed.pack(side="left", padx=2)
        self.drag = tk.BooleanVar(value=mocap_id is not None)
        ttk.Checkbutton(opt, text="drag ball in viewer", variable=self.drag,
                        state=("normal" if mocap_id is not None else "disabled")
                        ).pack(side="left", padx=10)

        self.status = ttk.Label(box, text="idle", foreground="#555")
        self.status.pack(side="top", anchor="w", padx=4, pady=(0, 4))

        self.rebuild()

    # -- setup -------------------------------------------------------------------

    def rebuild(self) -> None:
        """Re-derive the solved joint set after the end effector changes."""
        try:
            self.chain = ArmChain(self.model, self.body_var.get(),
                                  tcp_offset=[v.get() for v in self.tcp_vars])
        except ValueError as exc:
            self.chain = None
            self.chain_lbl.config(text=str(exc))
            return
        self.q_ik = None
        self.chain_lbl.config(
            text=(f"{len(self.chain)} joint(s): {', '.join(self.chain.names)}"
                  if len(self.chain) else
                  "no hinge joints between this body and the world"))
        self.snap()

    def apply_tcp(self) -> None:
        if self.chain is None:
            return
        self.chain.tcp_offset = np.array([v.get() for v in self.tcp_vars],
                                         dtype=float)
        self.snap()

    def estimate(self) -> None:
        guess = estimate_tcp(self.model, self.data, self.body_var.get())
        if guess is None:
            self.status.config(text="no children to average - set the TCP by hand")
            return
        for v, g in zip(self.tcp_vars, guess):
            v.set(round(float(g), 4))
        self.apply_tcp()
        self.status.config(text="TCP estimated - check the coloured dot")

    def snap(self) -> None:
        """Put the target on the current TCP so Follow cannot lurch."""
        if self.chain is None:
            return
        pos = self.chain.ee_pos(self.data)
        for v, p in zip(self.tgt_vars, pos):
            v.set(round(float(p), 3))
        if self.mocap_id is not None:
            self.data.mocap_pos[self.mocap_id] = pos

    # -- per frame ---------------------------------------------------------------

    def target(self) -> np.ndarray:
        """Current target, from the dragged ball or the typed boxes."""
        if self.mocap_id is not None and self.drag.get():
            pos = np.array(self.data.mocap_pos[self.mocap_id], dtype=float)
            for v, p in zip(self.tgt_vars, pos):
                v.set(round(float(p), 3))
            return pos
        pos = np.array([v.get() for v in self.tgt_vars], dtype=float)
        if self.mocap_id is not None:
            self.data.mocap_pos[self.mocap_id] = pos
        return pos

    def step(self) -> None:
        """One IK update, if this panel is active."""
        if not self.active.get() or self.chain is None or len(self.chain) == 0:
            return
        if self.q_ik is None:
            self.q_ik = self.data.qpos[self.chain.qadr].copy()
        previous = self.q_ik.copy()
        self.q_ik, err = self.chain.solve(self.data, self.target(),
                                          seed=self.q_ik, iters=ITERS_PER_FRAME)

        # Slew limit, scaling every joint by the SAME factor so the arm keeps
        # following the intended path instead of distorting toward whichever
        # joint saturated first.
        now = time.monotonic()
        dt = min(max(now - self._last_t, 1e-4), 0.1)      # ignore pauses
        self._last_t = now
        try:
            cap = math.radians(float(self.speed.get())) * dt
        except ValueError:
            cap = math.radians(DEFAULT_MAX_SPEED_DPS) * dt
        delta = self.q_ik - previous
        biggest = float(np.max(np.abs(delta))) if delta.size else 0.0
        if biggest > cap > 0:
            self.q_ik = previous + delta * (cap / biggest)

        self.chain.command(self.data, self.q_ik)
        self.status.config(text=f"error {err * 1000:6.1f} mm")

    def tcp_pos(self):
        return None if self.chain is None else self.chain.ee_pos(self.data)


def warn_if_shared(panels, label_widget) -> None:
    """Flag joints claimed by more than one active panel.

    Two solvers commanding the same joint overwrite each other every frame, so
    neither converges and the arm oscillates. It looks like a tuning problem and
    is not one, so it is worth naming explicitly.
    """
    active = [p for p in panels if p.active.get() and p.chain is not None]
    seen: dict[str, str] = {}
    clashes = []
    for panel in active:
        for name in panel.chain.names:
            if name in seen:
                clashes.append(f"{name} (arms {seen[name]} and {panel.label})")
            else:
                seen[name] = panel.label
    if clashes:
        label_widget.config(
            foreground="#b71c1c",
            text="SHARED JOINTS - the two solvers will fight: " +
                 ", ".join(clashes[:4]))
    else:
        label_widget.config(foreground="#555",
                            text="chains are independent" if len(active) > 1
                                 else "")