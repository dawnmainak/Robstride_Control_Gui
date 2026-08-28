# `mujoco_ik_dashboard.py` — teleop control

The main control surface. Drive the robot by its end effectors, record motion,
replay it, and optionally push the same commands to real motors.

```bash
pip install mujoco numpy          # plus python3-tk on Linux
python mujoco/3_teleop/mujoco_ik_dashboard.py mujoco/1_build_scene/scene.xml
```

It needs a built `scene.xml`. The original `mujoco_dashboard.py` is untouched
and still works; keep it as a fallback.

| File | Purpose |
|---|---|
| `mujoco_ik_dashboard.py` | The UI: four tabs, the control loop, the output limiter |
| `ik.py` | Damped least-squares IK, chain derivation, tool centre point |
| `ik_panel.py` | One reusable IK panel; the dashboard builds two |
| `motion.py` | Trajectory recorder and multi-track player |

---

## Read this before connecting hardware

The sim bridge's *default* joint map is `revolute_<can_id> -> <can_id>`, and
these joints are named `revolute_N`. So `revolute_3` would silently bind to CAN
id 3 — a real motor — even though `revolute_3` is an RS-03 hip joint that may be
nothing to do with your physical motor 3.

Never rely on the default. Write `sim_bindings.json` explicitly and list **only**
joints you have commissioned hardware for.

---

## The four tabs

Replacing the original single column, which did not fit on a laptop once a model
had twenty joints.

### Joints

One slider per actuator, trapezoidal **Move** / **Move ALL**, and **Capture zero
offsets**, which prints the current pose as a paste-ready block.

**SPEED LIMIT (deg/s)** is the global ceiling, and the important control on this
tab. It applies to every joint and every source — typed positions, dragged
sliders, IK and replayed takes alike — because it runs **last**, immediately
before the sim advances:

```python
previous_ctrl = np.array(d.ctrl[:nu])   # top of frame
...                                     # every source writes d.ctrl freely
for i in range(nu):                     # then walk it back at the allowed speed
    delta = d.ctrl[i] - previous_ctrl[i]
    if abs(delta) > cap:
        d.ctrl[i] = previous_ctrl[i] + math.copysign(cap, delta)
```

Before this existed there were three separate limiters (trapezoid, IK, replay)
and one hole: dragging a slider wrote `d.ctrl` directly and the joint jumped in a
single frame. Measured with the limit at 45 deg/s, a slider slammed to its far
end now moves 0.126° per frame instead of arriving instantly.

Per-joint overrides live in `JOINT_MAX_SPEED_DPS` at the top of the file; the
default is `DEFAULT_JOINT_SPEED_DPS`.

Note this limits each joint **independently**, unlike the IK panels which scale a
whole chain together to preserve its Cartesian path. This is a safety ceiling,
not a trajectory shaper — so keep it *above* your normal IK speed, or IK paths
will bend when a joint hits it.

### End effector

Two independent IK panels, arms A and B.

1. Pick the end effector — every body is listed, leaves first. A leaf-only list
   would hide the gripper palm, which has the fingers as children and is usually
   what you want to drive.
2. **Estimate** → **Apply** puts the tool centre point between the fingertips.
3. Tick **Follow**.
4. Drag the **red** ball for A, the **blue** ball for B: double-click to select,
   Ctrl + right-drag to move, add Shift for depth.

The TCP dots are colour-matched to their target balls. Each arm has its own Max
speed field, so one can run slowly while the other holds.

**The TCP is a point, not a body origin.** A gripper's origin sits at the wrist;
the point you want to place is the grasp centre. IK uses `mj_jac` rather than
`mj_jacBody` — the Jacobian of an arbitrary point rigidly attached to a body — so
the TCP is adjustable live with **no change to the model XML**, and nothing an
operator does here can alter the model that drives real hardware.

**Estimate** averages the positions of the body's direct children. It is a
starting point, not a measurement, and it uses the *current* finger pose —
estimate with the fingers neutral, not curled, then refine against the dot.

**Targets are mocap bodies injected at load time** with `MjSpec`. The viewer can
only move bodies that exist in the model, and only mocap bodies move freely.
Injecting them at load rather than editing `scene.xml` means the generated file
stays untouched. Both carry `contype=0 conaffinity=0` — visual only, and they can
never collide with the robot they guide. Needs MuJoCo 3.2+; older versions
disable the drag toggles and fall back to typed targets.

**Two solvers must not share a joint.** They would overwrite each other every
frame and the arm oscillates — which looks exactly like a tuning problem and is
not one. `warn_if_shared` checks every frame and names the offending joints in
red; otherwise it confirms `chains are independent`. **Check that line before
enabling Follow on both.**

For this robot the two limb chains are disjoint, both attaching to the torso
through fixed joints:

- **Limb A** — `revolute_1, 2, 3, 4, 8, 9, 5, 6`
- **Limb B** — `revolute_7, 12, 13, 14, 15, 16, 17, 18`

If an actuated waist is added later, both chains would include it and the warning
would fire; drive the waist from the Joints tab and pick end effectors below it.

Unreachable targets report honest residual error in mm rather than pretending —
a large steady number means out of reach or a joint at its limit.

### Motion

A mouse can only drag one target at a time, so record each arm separately:

1. **Record A**, drag the red ball, **Stop**
2. **Record B**, drag the blue ball, **Stop**
3. **Play both** — loop on by default

Each recorder captures **only the joints of its own IK chain**, so the two takes
own disjoint joints and merging them cannot conflict. Playback runs both off one
clock, so they are simultaneous by construction rather than by being started
together and hoping. Verified with two counter-moving joints: their sum stayed at
exactly 0.0000 throughout replay.

**Joint angles are recorded, not TCP positions.** Replaying angles reproduces the
take exactly; replaying Cartesian targets would re-run IK, and the solver can
pick a different arm configuration for the same point — elbow up instead of down.

Samples at 20 Hz, interpolated on playback so a recording does not become
stair-stepping on a motor. Takes of different length loop over the longer one,
with the shorter arm holding its final pose until the cycle restarts — looping
each on its own period would let the arms drift further out of phase on every
repeat. Save/Load per take, so one arm-A take can be paired with several arm-B
takes.

Replay overrides both sliders and IK: it is reproducing a take, and letting
either edit it mid-playback would corrupt it. Stop playback to get manual control
back.

### Motor

Streaming to hardware and mirroring hardware back, both **OFF at launch** and on
their own tab, so a whole motion can be rehearsed before anything is energised.

**Mirror mode** drives the model from measured angles and skips the physics step,
making it a viewer of the real robot rather than a simulation of it. Use it with
motors *disabled*, back-drive a joint by hand, and check the model follows — that
verifies a binding and its direction sign without energising anything. Do not run
mirror and streaming together: the sim would show where the motor is while
telling it to go somewhere else, with a network delay in the loop.

**Set zero from real** records each joint's measured angle as an offset, so from
then on a sim angle of 0 commands the pose the robot is in right now:

```
motor command  = sim angle + offset
mirrored angle = measured  - offset
```

Applied live — no rebuild, and `scene.xml` is untouched. Saved to
`sim_zero_offsets.json`, so a zero survives restarting the dashboard.

It needs telemetry, which means **streaming on and the motors enabled**: the
bridge only replies with state when it receives a packet, and the worker only
reports status for enabled motors. With nothing reporting there is nothing to
capture, and the button says so in red rather than silently doing nothing.

To zero with the shaft free to move by hand, use **Set Zero** in the main GUI's
motor panel instead — that writes the motor's own mechanical zero to flash and
works while disabled. The two are complementary: one changes the motor's zero,
the other changes the sim-to-motor mapping.

---

## Two findings worth keeping

Both present as "the IK is broken" when neither is an IK fault.

**IK iterates the commanded configuration, not the measured one.** Re-seeding the
solver from live `qpos` each frame makes the command lead actual position by only
one small step, so the arm creeps at whatever speed the servos happen to track —
measured at **110 mm of residual after three seconds**. Passing the previous
solution as `seed` converges in the solver while the servos are still moving.

**IK slews far faster than a motor should move.** Unclamped, a single frame moved
a joint **22.9°**, and at ~1900 solves/s that is a **43,000 deg/s** setpoint going
straight to hardware. The slider path was protected by its trapezoid profiles; IK
had no equivalent. Each panel now has its own Max speed field, and every joint is
scaled by the same factor so the arm keeps its intended path instead of
distorting toward whichever joint saturated.

---

## Safety

The sim can command anything its joint limits allow, so the model's joint ranges
are the safety envelope for the real robot — not a cosmetic detail.

Everything in `REMOTE.md` applies when driving real motors: the 1.5 s link
watchdog, the motor's `canTimeout`, and a physical E-stop wired to motor power.

Commissioning order that works:

1. Rehearse in sim with the Motor tab untouched
2. Bind **one** joint in `sim_bindings.json`
3. Enable Mirror with the motor **disabled**, back-drive by hand, confirm the
   direction sign
4. Enable streaming at 10 deg/s, with a hand on the power switch
5. Only then add the next joint

A wrong binding on one joint is a surprise; on sixteen it is a wrecked robot.

Replay is the case to be most careful with: with streaming on it drives the real
robot at the recorded speed, and the merged two-arm loop is the first time the
arms move together — which is also the first chance they have to collide with
each other. Watch that in sim before enabling hardware.

---

## Known gaps

- **`sim_bindings.json` is not fully populated.** Dual-arm IK on hardware needs
  all 16 driven joints; a partial map means the sim reaches a target using joints
  the hardware does not have.
- **A joint below the end effector is not IK-controlled.** Wrists and grippers
  sit outside the chain and are driven from the Joints tab. That is correct, but
  it is the reason a slider can appear to do nothing while Follow is on.
- **The gripper followers have no sliders** — the coupling removes their
  actuators. Drive `revolute_10` and `revolute_19`; their partners follow. Eighteen
  sliders for twenty joints is expected.
- **No collision geometry exists.** Every link is visual-only, so contact against
  the pedestal or between the arms is only meaningful once collision shapes are
  added.