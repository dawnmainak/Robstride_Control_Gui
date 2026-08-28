#!/usr/bin/env python3
"""Teleop dashboard: joint sliders, end-effector IK, sequences, live plots.

    python mujoco_ik_dashboard.py [scene.xml]

A tabbed rework of ``mujoco_dashboard.py`` with one new capability: pick an end
effector, set a target point, and the arm solves for it - so an operator places
a hand in space instead of coordinating six sliders by hand.

The original ``mujoco_dashboard.py`` is untouched and still works; keep it as a
fallback while this one is being trusted.

Layout is a notebook rather than one long column, because the old single stack
of controls did not fit on a laptop screen once a model had twenty joints:

  Joints    per-joint sliders, trapezoid moves, Move ALL
  End eff.  IK - choose the body, set a target, Go / Follow
  Sequence  capture poses, loop them
  Motor     stream to real hardware, mirror hardware back into the sim

Safety: streaming to real motors is OFF at launch on every tab, and IK Follow
is separate from streaming - you can rehearse a whole motion in sim and only
then turn the hardware on.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import os
import socket
import sys
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco is required:  pip install mujoco")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ik import candidate_bodies                      # noqa: E402
from ik_panel import IKPanel, warn_if_shared         # noqa: E402
from motion import MotionRecorder, MultiPlayer       # noqa: E402

# --- motor bridge ---------------------------------------------------------------
GUI_HOST, GUI_PORT = "127.0.0.1", 8642
MOTOR_HZ = 100.0
_mot_period = 1.0 / MOTOR_HZ
_mot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_mot_sock.setblocking(False)                  # never stall the sim on the network
_last_mot_send = 0.0
last_meas: dict[str, dict] = {}

#: Per-joint zero correction, radians, keyed by joint name. The value is the
#: MEASURED motor angle that corresponds to sim zero, so:
#:     motor command  = sim angle + offset
#:     mirrored angle = measured  - offset
#: Captured at runtime from the real robot; nothing is rebuilt and scene.xml is
#: untouched. Persisted so a zero survives restarting the dashboard.
zero_offset: dict[str, float] = {}

ZERO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "sim_zero_offsets.json")


def load_zero_offsets() -> None:
    try:
        with open(ZERO_FILE) as fh:
            for name, value in json.load(fh).items():
                zero_offset[str(name)] = float(value)
    except (OSError, ValueError, TypeError):
        pass                            # no file yet, or unreadable - start clean


def save_zero_offsets() -> None:
    try:
        with open(ZERO_FILE, "w") as fh:
            json.dump(zero_offset, fh, indent=2)
    except OSError:
        pass                            # a lost zero is an annoyance, not a fault


load_zero_offsets()

#: IK iterations per frame. Small because the solution is carried between frames
#: (see the seed argument in ik.solve) - the solver converges over a few frames
#: while the servos are still moving, instead of being restarted each time.
IK_ITERS_PER_FRAME = 4


def motor_send(targets_rad: dict) -> None:
    try:
        _mot_sock.sendto(json.dumps({"targets": targets_rad}).encode(),
                         (GUI_HOST, GUI_PORT))
    except OSError:
        pass


def motor_drain():
    latest = None
    while True:
        try:
            data, _ = _mot_sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            break
        try:
            st = json.loads(data.decode()).get("state")
            if isinstance(st, dict):
                latest = st
        except ValueError:
            pass
    return latest


class Trap:
    """Trapezoidal position profile: accelerate, cruise, decelerate."""

    def __init__(self, q0, q1, vmax, acc):
        self.q0, self.q1 = float(q0), float(q1)
        d = abs(self.q1 - self.q0)
        self.sign = 1.0 if self.q1 >= self.q0 else -1.0
        vmax, acc = max(float(vmax), 1e-6), max(float(acc), 1e-6)
        if d < 1e-9:
            self.ta = self.tc = self.T = 0.0
            return
        if vmax * vmax / acc >= d:            # triangular: never reaches vmax
            self.ta = math.sqrt(d / acc)
            self.tc = 0.0
            self.vpeak = acc * self.ta
        else:
            self.ta = vmax / acc
            self.tc = (d - vmax * vmax / acc) / vmax
            self.vpeak = vmax
        self.T = 2 * self.ta + self.tc

    def pos(self, t):
        if self.T <= 0:
            return self.q1
        t = max(0.0, min(float(t), self.T))
        if t < self.ta:
            s = 0.5 * (self.vpeak / self.ta) * t * t
        elif t < self.ta + self.tc:
            s = 0.5 * self.vpeak * self.ta + self.vpeak * (t - self.ta)
        else:
            td = t - self.ta - self.tc
            s = (0.5 * self.vpeak * self.ta + self.vpeak * self.tc
                 + self.vpeak * td - 0.5 * (self.vpeak / self.ta) * td * td)
        return self.q0 + self.sign * s


# --- model ----------------------------------------------------------------------
_DEFAULT_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "1_build_scene", "scene.xml")
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_SCENE

#: Draggable IK targets injected at load time - one per arm, coloured to match
#: the TCP dots so it is obvious which ball drives which arm.
TARGET_BODIES = {"A": (0.95, 0.25, 0.25, 0.6), "B": (0.25, 0.45, 0.95, 0.6)}


def load_model_with_targets(path):
    """Load the scene and add one draggable mocap ball per arm.

    The viewer can only move bodies that exist in the model, and only mocap
    bodies move freely - a normal body is pinned by its joints. Rather than
    require an edit to scene.xml, which urdf_to_scene.py regenerates, the balls
    are injected here with MjSpec: any scene gains drag handles and the file on
    disk stays exactly as generated.

    Returns (model, {label: mocap_id}). The map is empty when MjSpec is
    unavailable (MuJoCo older than 3.2), and the typed target boxes still work.
    """
    if not hasattr(mujoco, "MjSpec"):
        return mujoco.MjModel.from_xml_path(path), {}
    try:
        spec = mujoco.MjSpec.from_file(path)
        for label, rgba in TARGET_BODIES.items():
            body = spec.worldbody.add_body()
            body.name = f"ik_target_{label}"
            body.mocap = True                  # free to drag, ignores physics
            body.pos = [0.0, 0.0, 0.0]
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            geom.size = [0.02, 0.0, 0.0]
            geom.rgba = list(rgba)
            geom.contype = 0                   # visual only: never collides with
            geom.conaffinity = 0               # the robot it is guiding
        model = spec.compile()
        return model, {label: int(model.body(f"ik_target_{label}").mocapid[0])
                       for label in TARGET_BODIES}
    except Exception as exc:                   # keep working without handles
        print(f"could not add drag targets ({exc}); typed targets only",
              file=sys.stderr)
        return mujoco.MjModel.from_xml_path(path), {}


m, MOCAP_IDS = load_model_with_targets(MODEL_PATH)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

info = []
for a in range(m.nu):
    j = int(m.actuator_trnid[a][0])
    info.append({
        "a": a,
        "joint": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
        "q": int(m.jnt_qposadr[j]),
        "v": int(m.jnt_dofadr[j]),
        "lo": float(m.jnt_range[j][0]),
        "hi": float(m.jnt_range[j][1]),
    })
nu = len(info)
if nu == 0:
    sys.exit("this model has no actuators - run urdf_to_scene.py first")

# --- window ---------------------------------------------------------------------
root = tk.Tk()
root.title(f"RobStride teleop  -  {os.path.basename(MODEL_PATH)}  ({nu} joints)")
style = ttk.Style()
try:
    style.theme_use("clam")
except tk.TclError:
    pass
style.configure("Danger.TCheckbutton", foreground="#b71c1c")
style.configure("Head.TLabel", font=("TkDefaultFont", 10, "bold"))

nb = ttk.Notebook(root)
nb.pack(side="left", fill="both", expand=True, padx=6, pady=6)

tab_joints = ttk.Frame(nb)
tab_ik = ttk.Frame(nb)
tab_seq = ttk.Frame(nb)
tab_motor = ttk.Frame(nb)
for frame, label in ((tab_joints, "Joints"), (tab_ik, "End effector"),
                     (tab_seq, "Motion"), (tab_motor, "Motor")):
    nb.add(frame, text=label)

status = ttk.Label(root, text="ready", relief="sunken", anchor="w")
status.pack(side="bottom", fill="x")

# --- Joints tab -----------------------------------------------------------------
gp = ttk.Frame(tab_joints)
gp.pack(side="top", fill="x", pady=4)
ttk.Label(gp, text="Max speed (deg/s)").pack(side="left")
e_vmax = ttk.Entry(gp, width=6)
e_vmax.insert(0, "30")
e_vmax.pack(side="left", padx=(2, 8))
ttk.Label(gp, text="Accel").pack(side="left")
e_acc = ttk.Entry(gp, width=6)
e_acc.insert(0, "60")
e_acc.pack(side="left", padx=2)

profiles = [None] * nu
prof_t0 = [0.0] * nu
sliders, entries = [], []


def params():
    try:
        return max(float(e_vmax.get()), 1e-6), max(float(e_acc.get()), 1e-6)
    except ValueError:
        return 30.0, 60.0


def start(i):
    try:
        target = float(entries[i].get())
    except ValueError:
        return
    vmax, acc = params()
    profiles[i] = Trap(sliders[i].get(), target, vmax, acc)
    prof_t0[i] = d.time


def move_all():
    for i in range(nu):
        start(i)


jwrap = ttk.Frame(tab_joints)
jwrap.pack(side="top", fill="both", expand=True)
canvas = tk.Canvas(jwrap, width=430, highlightthickness=0)
scroll = ttk.Scrollbar(jwrap, orient="vertical", command=canvas.yview)
jlist = ttk.Frame(canvas)
# A twenty-joint model overflows any laptop screen, so the slider stack scrolls.
jlist.bind("<Configure>",
           lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=jlist, anchor="nw")
canvas.configure(yscrollcommand=scroll.set)
canvas.pack(side="left", fill="both", expand=True)
scroll.pack(side="right", fill="y")

for i in range(nu):
    lo_deg, hi_deg = math.degrees(info[i]["lo"]), math.degrees(info[i]["hi"])
    row = ttk.Frame(jlist)
    row.pack(side="top", fill="x", pady=1)
    ttk.Label(row, text=info[i]["joint"], width=14, anchor="w").pack(side="left")
    s = tk.Scale(row, from_=round(math.radians(lo_deg), 2),
                 to=round(math.radians(hi_deg), 2),
                 resolution=0.01, orient="horizontal", length=170,
                 showvalue=True)
    s.set(round(float(d.ctrl[i]), 2))
    s.pack(side="left")
    sliders.append(s)
    e = ttk.Entry(row, width=6)
    e.insert(0, "0")
    e.pack(side="left", padx=2)
    entries.append(e)
    ttk.Button(row, text="Move", width=6,
               command=lambda i=i: start(i)).pack(side="left")

def capture_zero():
    """Print a JOINT_ZERO_OFFSETS block for the current pose.

    Pose the sim until it matches the real robot at ITS zero, then press this.
    The values are the sim angles that correspond to real zero; urdf_to_scene.py
    turns them into MuJoCo ``ref`` attributes, inverting the sign for you -
    getting that backwards doubles the error instead of cancelling it.

    Only joints actually off zero are printed, so the block stays short.
    """
    lines = []
    for i in range(nu):
        deg = math.degrees(float(d.qpos[info[i]["q"]]))
        if abs(deg) >= 0.05:                 # ignore solver noise
            lines.append(f'    "{info[i]["joint"]}": {deg:.2f},')
    block = ("JOINT_ZERO_OFFSETS = {\n" + "\n".join(lines) + "\n}"
             if lines else "JOINT_ZERO_OFFSETS = {}   # already at zero")
    print("\n# paste into urdf_to_scene.py, then rebuild the scene")
    print(block, flush=True)
    status.config(text=f"zero offsets for {len(lines)} joint(s) printed "
                       f"to the terminal")


zrow = ttk.Frame(tab_joints)
zrow.pack(side="top", fill="x", pady=4)
ttk.Button(zrow, text="Move ALL", command=move_all).pack(side="left", padx=4)
ttk.Button(zrow, text="Capture zero offsets",
           command=capture_zero).pack(side="left", padx=4)
ttk.Label(tab_joints, wraplength=400, justify="left", foreground="#555",
          text="Capture zero: pose the sim to match the real robot at its zero, "
               "then press. The block is printed to the terminal for pasting "
               "into urdf_to_scene.py.").pack(side="top", anchor="w", padx=4)

# --- End effector tab -----------------------------------------------------------
ttk.Label(tab_ik, text="Inverse kinematics", style="Head.TLabel").pack(
    side="top", anchor="w", pady=(6, 2))
ttk.Label(tab_ik, wraplength=400, justify="left",
          text="Two arms, solved independently in the same frame. Tick Follow "
               "on each arm you want driven, then drag its coloured ball in the "
               "3D view or type a target. Only joints between the chosen body "
               "and the world are solved, so one arm does not disturb the "
               "other.").pack(side="top", anchor="w", pady=(0, 6))

# Two panels. Defaults aim at the two palms in the RobStride humanoid; on any
# other model they fall back to the first candidate body and the operator picks.
_palms = [b for b in candidate_bodies(m) if "palm" in b.lower()]
panels = [
    IKPanel(tab_ik, m, d, "A", MOCAP_IDS.get("A"),
            default_body=_palms[0] if len(_palms) > 0 else None),
    IKPanel(tab_ik, m, d, "B", MOCAP_IDS.get("B"),
            default_body=_palms[1] if len(_palms) > 1 else None),
]

shared_lbl = ttk.Label(tab_ik, wraplength=400, justify="left", foreground="#555")
shared_lbl.pack(side="top", anchor="w", pady=4)

ttk.Label(tab_ik, wraplength=400, justify="left", foreground="#555",
          text=("Double-click a ball to select it, then hold Ctrl and drag with "
                "the RIGHT mouse button to move it in the view plane; add Shift "
                "for depth. Red drives arm A, blue drives arm B."
                if MOCAP_IDS else
                "Viewer dragging needs MuJoCo 3.2 or newer - typed targets only.")
          ).pack(side="top", anchor="w", pady=(2, 6))

# --- Motion tab: record each arm separately, replay them together ----------------
ttk.Label(tab_seq, text="Record and replay motion", style="Head.TLabel").pack(
    side="top", anchor="w", pady=(6, 2))
ttk.Label(tab_seq, wraplength=400, justify="left",
          text="A mouse can only drag one target at a time, so record each arm "
               "on its own: Record A and move arm A, then Record B and move arm "
               "B. Play runs both takes off ONE clock, so the arms move "
               "together even though they were never performed together.").pack(
                   side="top", anchor="w", pady=(0, 8))

# One recorder per arm. Each captures ONLY the joints of its own IK chain, so
# the two takes own disjoint joints and merging them cannot conflict.
tracks = [MotionRecorder(), MotionRecorder()]
player = MultiPlayer(tracks)


def _chain_joints(index: int) -> list:
    panel = panels[index]
    if panel.chain is None:
        return []
    return list(panel.chain.names)


def make_track_row(index: int, label: str):
    row = ttk.LabelFrame(tab_seq, text=f"Arm {label} take")
    row.pack(side="top", fill="x", pady=3, padx=2)
    inner = ttk.Frame(row)
    inner.pack(side="top", fill="x", padx=4, pady=3)

    def toggle():
        rec = tracks[index]
        if rec.recording:
            rec.stop()
            btn.config(text=f"Record {label}")
            info_lbl.config(foreground="#555",
                            text=f"{len(rec.samples)} samples, {rec.duration:.1f}s")
            return
        joints = _chain_joints(index)
        if not joints:
            info_lbl.config(foreground="#b71c1c",
                            text="pick an end effector for this arm first")
            return
        player.stop()
        play_btn.config(text="Play both")
        for other in tracks:                 # only one take at a time
            other.stop()
        for other_btn, other_label in buttons:
            other_btn.config(text=f"Record {other_label}")
        rec.start()
        btn.config(text="Stop")
        info_lbl.config(foreground="#b71c1c",
                        text=f"RECORDING {len(joints)} joint(s)")

    btn = ttk.Button(inner, text=f"Record {label}", command=toggle)
    btn.pack(side="left")
    path_entry = ttk.Entry(inner, width=16)
    path_entry.insert(0, f"take_{label}.json")
    path_entry.pack(side="left", padx=6)

    def save():
        try:
            n = tracks[index].save(path_entry.get().strip() or f"take_{label}.json")
            info_lbl.config(foreground="#2e7d32", text=f"saved {n} samples")
        except OSError as exc:
            info_lbl.config(foreground="#b71c1c", text=f"save failed: {exc}")

    def load():
        try:
            n = tracks[index].load(path_entry.get().strip() or f"take_{label}.json")
            info_lbl.config(foreground="#555",
                            text=f"loaded {n} samples, {tracks[index].duration:.1f}s")
        except (OSError, ValueError) as exc:
            info_lbl.config(foreground="#b71c1c", text=f"load failed: {exc}")

    def clear():
        tracks[index].samples = []
        info_lbl.config(foreground="#555", text="cleared")

    ttk.Button(inner, text="Save", width=6, command=save).pack(side="left", padx=2)
    ttk.Button(inner, text="Load", width=6, command=load).pack(side="left", padx=2)
    ttk.Button(inner, text="Clear", width=6, command=clear).pack(side="left", padx=2)
    info_lbl = ttk.Label(row, text="empty", foreground="#555")
    info_lbl.pack(side="top", anchor="w", padx=6, pady=(0, 4))
    return btn, label


buttons = []
buttons.append(make_track_row(0, "A"))
buttons.append(make_track_row(1, "B"))

play_row = ttk.Frame(tab_seq)
play_row.pack(side="top", fill="x", pady=8)


def toggle_play():
    if player.playing:
        player.stop()
        play_btn.config(text="Play both")
        motion_status.config(foreground="#555", text="stopped")
        return
    if player.duration <= 0:
        motion_status.config(foreground="#b71c1c", text="nothing recorded yet")
        return
    for rec in tracks:
        rec.stop()
    for b, lab in buttons:
        b.config(text=f"Record {lab}")
    try:
        player.speed = max(0.1, float(speed_entry.get()))
    except ValueError:
        player.speed = 1.0
    player.loop = loop_var.get()
    player.start()
    play_btn.config(text="Stop")


play_btn = ttk.Button(play_row, text="Play both", command=toggle_play)
play_btn.pack(side="left")
loop_var = tk.BooleanVar(value=True)
ttk.Checkbutton(play_row, text="loop", variable=loop_var).pack(side="left", padx=8)
ttk.Label(play_row, text="speed").pack(side="left")
speed_entry = ttk.Entry(play_row, width=5)
speed_entry.insert(0, "1.0")
speed_entry.pack(side="left", padx=2)

progress = ttk.Progressbar(tab_seq, mode="determinate", maximum=100)
progress.pack(side="top", fill="x", pady=6)
motion_status = ttk.Label(tab_seq, text="idle", foreground="#555")
motion_status.pack(side="top", anchor="w")

ttk.Label(tab_seq, wraplength=400, justify="left", foreground="#555",
          text="Takes of different lengths loop over the longer one; the "
               "shorter arm holds its final pose until the cycle restarts. "
               "Replay writes joint commands directly, so with streaming on it "
               "drives the real robot at the recorded speed.").pack(
                   side="top", anchor="w", pady=8)

# --- Motor tab ------------------------------------------------------------------
ttk.Label(tab_motor, text="Real hardware", style="Head.TLabel").pack(
    side="top", anchor="w", pady=(6, 2))
ttk.Label(tab_motor, wraplength=400, justify="left",
          text="Both are OFF at launch so nothing moves when this window opens. "
               "The GUI must be running with its sim bridge enabled, and only "
               "joints listed in sim_bindings.json are driven.").pack(
                   side="top", anchor="w", pady=(0, 8))

mot_var = tk.BooleanVar(value=False)
ttk.Checkbutton(tab_motor, text="Stream commands to real motors",
                variable=mot_var, style="Danger.TCheckbutton").pack(
                    side="top", anchor="w")
mirror_var = tk.BooleanVar(value=False)
ttk.Checkbutton(tab_motor, text="Mirror real motors into the sim",
                variable=mirror_var).pack(side="top", anchor="w")
ttk.Label(tab_motor, wraplength=400, justify="left", foreground="#555",
          text="Mirror drives the MODEL from measured angles and skips the "
               "physics step, so it is a viewer of the real robot. Use it with "
               "motors disabled to back-drive a joint by hand and confirm the "
               "binding and direction. Do not run both at once.").pack(
                   side="top", anchor="w", pady=6)
mot_status = ttk.Label(tab_motor, text="motor: (off)")
mot_status.pack(side="top", anchor="w", pady=4)

def set_zero_from_real():
    """Call the robot's CURRENT position sim-zero, for every reporting joint.

    Records the measured motor angle as that joint's offset, so from now on a
    sim angle of 0 commands the pose the robot is in right now. Applied live -
    no rebuild, and scene.xml is not touched.

    Needs telemetry, which means streaming ON and the motor ENABLED: the bridge
    only replies with state when it receives a packet, and the worker only
    reports status for enabled motors. With nothing reporting there is nothing
    to capture, so that case is reported rather than silently doing nothing.
    """
    if not last_meas:
        zero_status.config(
            foreground="#b71c1c",
            text="no telemetry - turn streaming on and enable the motors first")
        return
    taken = []
    for i in range(nu):
        name = info[i]["joint"]
        measured = last_meas.get(name)
        if not measured:
            continue
        pos = measured.get("pos")
        if isinstance(pos, (int, float)) and math.isfinite(pos):
            zero_offset[name] = float(pos)
            # Move the sim to the new zero too, so the model and the robot agree
            # the instant the button is pressed instead of on the next command.
            d.qpos[info[i]["q"]] = 0.0
            d.ctrl[i] = 0.0
            sliders[i].set(0.0)
            taken.append(name)
    mujoco.mj_forward(m, d)
    save_zero_offsets()
    for panel in panels:                 # targets were computed in the old frame
        panel.snap()
    zero_status.config(
        foreground="#2e7d32",
        text=f"zeroed {len(taken)} joint(s): {', '.join(taken) or 'none'}")


def clear_zero_offsets():
    zero_offset.clear()
    save_zero_offsets()
    zero_status.config(foreground="#555", text="zero offsets cleared")


ttk.Separator(tab_motor, orient="horizontal").pack(side="top", fill="x", pady=8)
ttk.Label(tab_motor, text="Zeroing", style="Head.TLabel").pack(
    side="top", anchor="w")
ttk.Label(tab_motor, wraplength=400, justify="left",
          text="Move the robot to the pose you want to call zero, then press "
               "Set zero from real. From then on a sim angle of 0 commands that "
               "pose. Needs streaming on and the motors enabled, since that is "
               "the only time position is reported back.").pack(
                   side="top", anchor="w", pady=(2, 6))
zrow2 = ttk.Frame(tab_motor)
zrow2.pack(side="top", fill="x")
ttk.Button(zrow2, text="Set zero from real",
           command=set_zero_from_real).pack(side="left", padx=(0, 6))
ttk.Button(zrow2, text="Clear zeros",
           command=clear_zero_offsets).pack(side="left")
zero_status = ttk.Label(
    tab_motor, wraplength=400, justify="left", foreground="#555",
    text=(f"{len(zero_offset)} saved offset(s) loaded" if zero_offset
          else "no zero offsets set"))
zero_status.pack(side="top", anchor="w", pady=6)

log_var = tk.BooleanVar(value=True)
ttk.Checkbutton(tab_motor, text="Log to CSV", variable=log_var).pack(
    side="top", anchor="w", pady=(12, 0))

# --- live buffers ---------------------------------------------------------------
N = 600
tbuf = collections.deque(maxlen=N)
posb = [collections.deque(maxlen=N) for _ in range(nu)]
velb = [collections.deque(maxlen=N) for _ in range(nu)]
torb = [collections.deque(maxlen=N) for _ in range(nu)]

csv_file = open("teleop_log.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["t"] + [f"{c['joint']}_{k}" for c in info
                         for k in ("cmd", "pos_deg", "vel_dps", "torque")])


#: Radius of the TCP / target markers, metres. Big enough to see against a
#: gripper, small enough not to hide it.
MARKER_R = 0.012


def draw_tcps(viewer, panels) -> None:
    """Draw each arm's TCP into the viewer, colour-matched to its target ball.

    Uses the viewer's user scene rather than adding sites to the model, so a TCP
    can be moved from the UI without regenerating scene.xml - and nothing the
    operator does here can alter the model that drives real hardware.
    """
    scn = getattr(viewer, "user_scn", None)
    if scn is None:                       # older viewer without a user scene
        return
    scn.ngeom = 0
    colours = {"A": (0.95, 0.35, 0.35, 1.0), "B": (0.35, 0.55, 0.95, 1.0)}
    for panel in panels:
        pos = panel.tcp_pos()
        if pos is None or scn.ngeom >= scn.maxgeom:
            continue
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([MARKER_R, 0.0, 0.0]),
            np.asarray(pos, dtype=float), np.eye(3).flatten(),
            np.array(colours.get(panel.label, (0.1, 0.9, 0.3, 1.0)),
                     dtype=np.float32))
        scn.ngeom += 1


def step_ik() -> str:
    """Advance every active arm one IK step. Returns a one-line summary."""
    for panel in panels:
        panel.step()
    warn_if_shared(panels, shared_lbl)
    active = [p for p in panels if p.active.get() and p.chain is not None]
    if not active:
        return "IK idle"
    # Keep the joint sliders showing what IK asked for, so switching to the
    # Joints tab does not immediately fight the solution.
    for panel in active:
        if panel.q_ik is None:
            continue
        for joint, angle in zip(panel.chain.joints, panel.q_ik):
            act = panel.chain.actuator_of.get(joint)
            for i in range(nu):
                if info[i]["a"] == act:
                    sliders[i].set(round(float(angle), 2))
    return "IK: " + "   ".join(f"{p.label} {p.status.cget('text')}"
                               for p in active)


with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # --- motion record / replay ---
        for _idx, _rec in enumerate(tracks):
            if not _rec.recording:
                continue
            # Only this arm's chain joints. Recording every joint would make the
            # two takes overlap, and merging them at playback would let one arm's
            # take silently overwrite the other's.
            _owned = set(_chain_joints(_idx))
            _rec.capture({info[i]["joint"]: float(d.ctrl[i])
                          for i in range(nu) if info[i]["joint"] in _owned})
            motion_status.config(
                foreground="#b71c1c",
                text=f"RECORDING arm {'AB'[_idx]} - {len(_rec.samples)} samples, "
                     f"{_rec.duration:.1f}s")
        replayed = player.step() if player.playing else {}
        if replayed:
            progress["value"] = player.progress() * 100
            if not player.playing:
                play_btn.config(text="Play")
                motion_status.config(text="playback finished")

        # --- IK, or the sliders ---
        # IK owns only the joints in its active chains. EVERY other joint must
        # still follow its slider - this used to be an if/else, so switching IK
        # on silently froze every joint outside the chain (the fingers below the
        # end effector, and the whole of the other limb). Their sliders moved
        # and nothing happened, which reads as dead motors rather than as a
        # control-authority question.
        ik_owned = set()
        for panel in panels:
            if panel.active.get() and panel.chain is not None:
                ik_owned.update(a for a in panel.chain.actuator_of.values())

        for i in range(nu):
            if profiles[i] is not None:
                tt = d.time - prof_t0[i]
                sliders[i].set(round(float(profiles[i].pos(tt)), 2))
                if tt >= profiles[i].T:
                    profiles[i] = None
            if info[i]["joint"] in replayed:
                # Replay wins over both sliders and IK: it is reproducing a take,
                # and letting either edit it mid-playback would corrupt it.
                d.ctrl[i] = replayed[info[i]["joint"]]
                sliders[i].set(round(d.ctrl[i], 2))
            elif info[i]["a"] not in ik_owned:
                d.ctrl[i] = sliders[i].get()

        if ik_owned and not replayed:
            status.config(text=step_ik())      # overwrites only its own joints

        # --- hardware ---
        now = time.monotonic()
        if mot_var.get() and (now - _last_mot_send) >= _mot_period:
            _last_mot_send = now
            # Shift each command into the MOTOR's frame before it leaves.
            motor_send({info[i]["joint"]:
                        float(d.ctrl[i]) + zero_offset.get(info[i]["joint"], 0.0)
                        for i in range(nu)})
            st = motor_drain()
            if st:
                last_meas.update(st)
            shown = ", ".join(f"{n}={v.get('pos', float('nan')):+.3f}"
                              for n, v in last_meas.items())
            mot_status.config(text=f"motor: {shown or 'no reply'}")
        elif not mot_var.get():
            mot_status.config(text="motor: (off)")

        # --- advance ---
        if mirror_var.get() and last_meas:
            # Write measured angles into qpos and recompute kinematics ONLY.
            # mj_step is skipped deliberately: stepping the integrator while
            # overwriting qpos each frame makes the actuators fight the values
            # just written and the model judders.
            for i in range(nu):
                mv = last_meas.get(info[i]["joint"])
                if not mv:
                    continue
                pos = mv.get("pos")
                if isinstance(pos, (int, float)) and math.isfinite(pos):
                    # Back into the SIM's frame - the inverse of the shift
                    # applied when commanding.
                    d.qpos[info[i]["q"]] = \
                        float(pos) - zero_offset.get(info[i]["joint"], 0.0)
            mujoco.mj_forward(m, d)
            d.time += m.opt.timestep         # keep the plot time axis moving
        else:
            mujoco.mj_step(m, d)

        # The targets are real bodies when drag handles exist, so only the TCPs
        # need synthetic markers.
        draw_tcps(viewer, panels)
        viewer.sync()

        tbuf.append(d.time)
        for i in range(nu):
            posb[i].append(math.degrees(d.qpos[info[i]["q"]]))
            velb[i].append(math.degrees(d.qvel[info[i]["v"]]))
            torb[i].append(d.qfrc_actuator[info[i]["v"]])
        if log_var.get():
            writer.writerow(
                [round(d.time, 4)] +
                [x for c in info for x in
                 (round(d.ctrl[c["a"]], 3),
                  round(math.degrees(d.qpos[c["q"]]), 3),
                  round(math.degrees(d.qvel[c["v"]]), 3),
                  round(d.qfrc_actuator[c["v"]], 4))])

        root.update_idletasks()
        root.update()

csv_file.close()