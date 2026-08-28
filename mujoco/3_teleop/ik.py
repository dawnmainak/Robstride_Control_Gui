"""Damped least-squares inverse kinematics for a MuJoCo arm.

Give it an end-effector body and a target point; it returns joint angles that
put the end effector there. Used by the teleop dashboard so an operator can drag
a point in space instead of coordinating six sliders by hand.

Why damped least squares rather than the plain Jacobian pseudo-inverse: near a
singularity (arm fully stretched, or two axes lined up) the pseudo-inverse
demands enormous joint velocities for a tiny Cartesian motion, and the arm
snaps. The damping term trades a little accuracy for a bounded step, which is
the right trade when the same angles are about to be sent to real hardware.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:                                   # pragma: no cover
    raise SystemExit("mujoco is required:  pip install mujoco")

#: Damping (lambda). Larger = steadier near singularities, slower to converge.
#: 0.05 keeps a metre-scale arm stable without feeling sluggish.
DEFAULT_DAMPING = 0.05

#: Fraction of the full solved step taken per iteration. Below 1 so the solution
#: eases toward the target rather than jumping - important when these angles go
#: to a real motor.
DEFAULT_STEP = 0.5

#: Cartesian error (metres) under which the target counts as reached.
DEFAULT_TOL = 1e-3

#: Largest joint change allowed in one solve, radians. A hard ceiling on top of
#: the damping: even a pathological Jacobian cannot produce a lurch bigger than
#: this, so a bad target degrades into slow motion instead of a snap.
MAX_STEP_RAD = 0.10


class ArmChain:
    """The hinge joints between the world and one end-effector body.

    Derived by walking up the body tree from the end effector, so only joints
    that can actually move it are solved for. On a 20-joint humanoid this means
    dragging the left hand does not stir the legs.
    """

    def __init__(self, model, ee_body: str, tcp_offset=None):
        self.model = model
        # Tool centre point, in the end-effector body's LOCAL frame. A gripper's
        # body origin sits at the wrist, but the point an operator wants to place
        # is the grasp centre between the fingertips. Solving for the body origin
        # puts the wrist on target and leaves the fingers somewhere else.
        self.tcp_offset = np.zeros(3) if tcp_offset is None \
            else np.asarray(tcp_offset, dtype=float)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise ValueError(f"no body named {ee_body!r} in this model")
        self.ee_name = ee_body

        joints: list[int] = []
        body = self.ee_id
        while body > 0:
            for j in range(model.body_jntadr[body],
                           model.body_jntadr[body] + model.body_jntnum[body]):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    joints.append(j)
            body = model.body_parentid[body]
        self.joints = list(reversed(joints))          # root-most first

        self.dofs = np.array([model.jnt_dofadr[j] for j in self.joints], dtype=int)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.lower = np.array([model.jnt_range[j][0] for j in self.joints])
        self.upper = np.array([model.jnt_range[j][1] for j in self.joints])
        self.limited = np.array([bool(model.jnt_limited[j]) for j in self.joints])
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                      for j in self.joints]

        # joint -> actuator, so a solution can be commanded rather than teleported
        self.actuator_of: dict[int, int] = {}
        for a in range(model.nu):
            if model.actuator_trnid[a][0] in self.joints:
                self.actuator_of[int(model.actuator_trnid[a][0])] = a

    def __len__(self) -> int:
        return len(self.joints)

    def ee_pos(self, data) -> np.ndarray:
        """World position of the TCP (body origin when the offset is zero)."""
        origin = np.array(data.xpos[self.ee_id])
        if not self.tcp_offset.any():
            return origin
        rot = np.array(data.xmat[self.ee_id]).reshape(3, 3)
        return origin + rot @ self.tcp_offset

    def _tcp_world(self, data) -> np.ndarray:
        origin = np.array(data.xpos[self.ee_id])
        rot = np.array(data.xmat[self.ee_id]).reshape(3, 3)
        return origin + rot @ self.tcp_offset

    def solve(self, data, target: np.ndarray, *,
              seed: "np.ndarray | None" = None,
              damping: float = DEFAULT_DAMPING,
              step: float = DEFAULT_STEP,
              iters: int = 8,
              tol: float = DEFAULT_TOL) -> tuple[np.ndarray, float]:
        """Joint angles that move the end effector toward ``target``.

        ``seed`` is the configuration to iterate FROM, and passing the previous
        solution is strongly preferred in a live loop. Re-seeding from the
        measured ``data.qpos`` every frame makes the command lead actual
        position by only one small step, so the arm creeps toward the target at
        the speed the servos happen to track rather than converging - measured
        at 110 mm of residual error after three seconds on a test arm. Iterating
        the commanded configuration instead converges in the solver, and the
        servos then follow a target that is already correct.

        Iterates on a scratch copy so ``data`` is never disturbed - the caller
        decides whether to command the result. Returns (angles, final error).
        """
        model = self.model
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        if seed is not None:
            scratch.qpos[self.qadr] = seed
        mujoco.mj_forward(model, scratch)

        jacp = np.zeros((3, model.nv))
        q = scratch.qpos[self.qadr].copy()
        err_norm = float("inf")

        for _ in range(iters):
            mujoco.mj_forward(model, scratch)
            point = self._tcp_world(scratch)
            err = np.asarray(target) - point
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                break

            # mj_jac (not mj_jacBody) gives the Jacobian of an ARBITRARY point
            # rigidly attached to the body, which is what makes the TCP offset
            # work with no change to the model XML. With a zero offset it is
            # identical to mj_jacBody.
            mujoco.mj_jac(model, scratch, jacp, None, point, self.ee_id)
            J = jacp[:, self.dofs]                     # 3 x n, this chain only

            # dq = J^T (J J^T + lambda^2 I)^-1 err
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)

            dq *= step
            largest = float(np.max(np.abs(dq))) if dq.size else 0.0
            if largest > MAX_STEP_RAD:
                dq *= MAX_STEP_RAD / largest           # scale, preserving direction

            q = q + dq
            if self.limited.any():
                q = np.where(self.limited, np.clip(q, self.lower, self.upper), q)
            scratch.qpos[self.qadr] = q

        return q, err_norm

    def command(self, data, angles: np.ndarray) -> None:
        """Write solved angles into ``data.ctrl`` for the chain's actuators.

        Commanding rather than writing qpos keeps the physics honest: the arm
        is driven to the pose by its actuators, exactly as the real one will be,
        so a pose the servos cannot hold shows up as a tracking error instead of
        looking perfect in sim and failing on hardware.
        """
        for joint, angle in zip(self.joints, angles):
            a = self.actuator_of.get(joint)
            if a is None:
                continue                                # joint has no actuator
            lo, hi = self.model.actuator_ctrlrange[a]
            if self.model.actuator_ctrllimited[a]:
                angle = float(np.clip(angle, lo, hi))
            data.ctrl[a] = angle


def leaf_bodies(model) -> list[str]:
    """Bodies with no children."""
    has_child = set(int(model.body_parentid[b]) for b in range(1, model.nbody))
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in range(1, model.nbody) if b not in has_child]


def candidate_bodies(model) -> list[str]:
    """Every body, leaves first, as end-effector choices.

    Leaves alone are not enough: a gripper PALM has the fingers as children, so
    a leaf-only list offers the fingertips but not the palm - and the palm is
    usually what you want to drive, with the TCP offset out to the grasp centre.
    """
    leaves = leaf_bodies(model)
    others = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
              for b in range(1, model.nbody)
              if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) not in leaves]
    return leaves + others


def estimate_tcp(model, data, body_name: str) -> "np.ndarray | None":
    """Guess a grasp-centre TCP as the mean of a body's children, in its frame.

    For a gripper palm whose children are the fingers, the average fingertip
    position is a decent first approximation of where an object would be held.
    A starting point to refine by eye, not a measurement. Returns None when the
    body has no children, where there is nothing to average.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    kids = [b for b in range(1, model.nbody) if int(model.body_parentid[b]) == bid]
    if not kids:
        return None
    centre = np.mean([data.xpos[b] for b in kids], axis=0)
    rot = np.array(data.xmat[bid]).reshape(3, 3)
    return rot.T @ (centre - np.array(data.xpos[bid]))     # world -> body frame