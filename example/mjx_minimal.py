"""
CIO for Panda Leap + Cylinder using MJX
"""
import time
import numpy as np
import jax.numpy as jnp
import mujoco as mj
import mujoco.mjx as mjx
from jaxopt import LBFGS
from collections import namedtuple
import os
from xvfbwrapper import Xvfb

from mjmanip.robot.panda_leap_mjx import PandaLeapMjxEnv, PandaLeapMjx, DEFAULT_SCENE_MJX_XML_PATH, ARM_XML_PATH, \
    HAND_XML_PATH
from video_recorder import VideoRecorder

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
Position = namedtuple('Position', 'x y z')
Pose = namedtuple('Pose', 'x y z quat')  # quat = [w, x, y, z]


class Params:
    def __init__(self, n_fingers, n_joints, K=10, delT=0.05, delT_phase=0.5,
                 mass=0.1, mu=0.5, lamb=1e-3):
        self.n_fingers = n_fingers
        self.n_joints = n_joints
        self.K = K
        self.delT = delT
        self.delT_phase = delT_phase
        self.mass = mass
        self.mu = mu
        self.lamb = lamb
        self.steps_per_phase = int(delT_phase / delT)
        self.T_steps = K * self.steps_per_phase
        self.T_final = K * delT_phase

        # Variables per phase:
        #   obj_pos (3) + obj_vel (3) + joint_angles (n_joints)
        #   + forces (3 * n_fingers) + contact_positions (3 * n_fingers)
        self.per_phase = 6 + n_joints + 6 * n_fingers
        self.len_s = self.per_phase
        self.len_S = self.per_phase * K


# ----------------------------------------------------------------------
# Load MuJoCo model and prepare MJX utilities
# ----------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Mujoco-based World class
# ------------------------------------------------------------------------------
class MujocoWorld:
    """
    Wraps a MuJoCo model to provide:
      - forward kinematics for fingertips given joint angles
      - object pose and velocity
      - collision distances and normals for contact points
    """

    def __init__(self, record_video: bool = True,
                 width: int = 640, height: int = 480):
        """
        """
        self.robot_env = PandaLeapMjxEnv(world_scene_xml=DEFAULT_SCENE_MJX_XML_PATH,
                                         arm_xml=ARM_XML_PATH,
                                         hand_xml=HAND_XML_PATH)
        spec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.01
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        m = self.model
        d = self.data

        # Cache body ids
        self.fingertip_ids = [self.model.body(name).id
                              for name in PandaLeapMjx.hand_items_full_names(PandaLeapMjx.FINGER_TIPS_BODY_NAMES)]
        self.object_id = self.model.body(PandaLeapMjx.OBJECT_NAMES[0]).id

        # Identify hand joints (excluding the object's free joint)
        joint_names = []
        joint_qposadr = []
        joint_limits = []
        n_joints = 0
        for jnt_id in range(m.njnt):
            jnt = m.joint(jnt_id)
            if jnt.type == mj.mjtJoint.mjJNT_FREE:
                continue
            joint_names.append(jnt.name)
            joint_qposadr.append(jnt.qposadr[0])
            joint_limits.append((jnt.range[0], jnt.range[1]))
            n_joints += 1

        # Convert to MJX model
        mjx_model = mjx.put_model(m, impl='warp')
        # Build MJX state (zero velocity)
        self.state = mjx.make_data(self.model, impl='warp', nconmax=100 * 8192, njmax=200)

        # Create a function that, given joint angles, returns fingertip positions
        # using MJX forward kinematics. We'll close over mjx_model, fingertip_ids, joint_qposadr.
        # Note: MJX state includes qpos for all joints. We need to know the indices
        # of the hand joints in the qpos array. They are stored in joint_qposadr.
        # We'll build a state with fixed object pose (the free joint comes first in qpos).
        # The object's free joint qpos indices: first 7 entries (pos+quat).
        # The hand joints follow in the order given by joint_qposadr.

        # Get the address of the free joint (assuming the object has one)
        obj_jnt_id = None
        for jnt_id in range(m.njnt):
            if m.joint(jnt_id).type == mj.mjtJoint.mjJNT_FREE:
                obj_jnt_id = jnt_id
                break
        if obj_jnt_id is None:
            raise ValueError("Object must have a free joint.")
        obj_qposadr = m.jnt_qposadr[obj_jnt_id]  # start index of free joint's qpos

        # Now define the fingertip position function
        def fingertip_pos_fn(joint_angles):
            # joint_angles: (n_joints,) array
            # Build full qpos: object fixed at origin (identity)
            qpos = jnp.zeros(m.nq)
            # Object free joint: position (0,0,0) and identity quat (1,0,0,0)
            qpos = qpos.at[obj_qposadr:obj_qposadr + 3].set(jnp.array([0., 0., 0.]))
            qpos = qpos.at[obj_qposadr + 3:obj_qposadr + 7].set(jnp.array([1., 0., 0., 0.]))

            # Set hand joints
            for i in range(joint_angles.shape[0]):
                qpos = qpos.at[jnp.array(joint_qposadr)].set(joint_angles[i])
                self.state = self.state.replace(qpos=qpos, qvel=jnp.zeros(m.nv))

                # Forward kinematics
                mjx.forward(mjx_model, self.state)

            # Extract fingertip positions
            pos = []
            for body_id in self.fingertip_ids:
                # body xpos is stored in state.xpos[body_id] (world position)
                pos.append(self.state.xpos[body_id])
            return jnp.stack(pos)  # (n_fingers, 3)

        self.info = {
            'mj_model': m,
            'mj_data': d,
            'mjx_model': mjx_model,
            'fingertip_ids': self.fingertip_ids,
            'object_id': self.object_id,
            'joint_qposadr': joint_qposadr,
            'joint_limits': jnp.array(joint_limits),
            'n_joints': n_joints,
            'n_fingers': len(self.fingertip_ids),
            'obj_qposadr': obj_qposadr,
            'fingertip_pos_fn': fingertip_pos_fn
        }

        # Initialize video recording if enabled
        self.mj_viewer: mj.viewer.Handle = None
        self.mj_renderer: mj.Renderer = None
        self.mj_recorder: VideoRecorder = None
        if record_video:
            # Create the video recorder
            self.mj_recorder = VideoRecorder(
                output_dir="recordings",
                width=width,
                height=height,
                fps=1 / 60,
            )
            # Ensure model visual offscreen buffer is compatible with video recording
            vis_global = self.model.vis.global_
            vis_global.offwidth = width
            vis_global.offheight = height
            self.mj_renderer = mj.Renderer(self.model, width=width, height=height)

    # ----------------------------------------------------------------------
    # Initial trajectory guess
    # ----------------------------------------------------------------------
    def init_trajectory(self, info, p, init_obj_pos, init_obj_vel):
        """Create initial guess: joints at qpos0, zero forces, contact positions at object center."""
        n_joints = info['n_joints']
        n_fingers = info['n_fingers']
        per_phase = p.per_phase

        # Get initial joint angles from MuJoCo data
        init_joints = jnp.array([info['mj_data'].qpos[adr] for adr in info['joint_qposadr']])

        S = np.zeros(p.len_S)
        for k in range(p.K):
            idx = k * per_phase
            # Object pos, vel
            S[idx:idx + 3] = init_obj_pos
            S[idx + 3:idx + 6] = init_obj_vel
            # Joint angles
            S[idx + 6:idx + 6 + n_joints] = init_joints
            # Forces (zero)
            force_start = idx + 6 + n_joints
            force_end = force_start + 3 * n_fingers
            S[force_start:force_end] = 0.0
            # Contact positions (object center)
            ro_start = force_end
            ro_end = ro_start + 3 * n_fingers
            S[ro_start:ro_end] = np.tile(init_obj_pos, n_fingers)
        return jnp.array(S)

    # ----------------------------------------------------------------------
    # Cost function (fully JAX‑compatible)
    # ----------------------------------------------------------------------
    def cost_function(self, S, p, goals, info):
        """
        S: flat array of all variables (K * per_phase)
        info: dictionary with model info, including fingertip_pos_fn
        """
        K = p.K
        per_phase = p.per_phase
        n_joints = p.n_joints
        n_fingers = p.n_fingers

        phases = S.reshape(K, per_phase)

        obj_pos = phases[:, 0:3]  # (K,3)
        obj_vel = phases[:, 3:6]  # (K,3)
        joint_angles = phases[:, 6:6 + n_joints]  # (K, n_joints)
        forces = phases[:, 6 + n_joints: 6 + n_joints + 3 * n_fingers].reshape(K, n_fingers, 3)
        ro = phases[:, 6 + n_joints + 3 * n_fingers:].reshape(K, n_fingers, 3)

        # Compute fingertip positions for all phases (vectorized)
        fingertip_pos = info['fingertip_pos_fn'](joint_angles)  # (K, n_fingers, 3)

        # ------------------------------------------------------------------
        # Contact invariant cost (L_CI)
        # ------------------------------------------------------------------
        # e_H: distance from fingertip to contact point (world frame)
        contact_world = obj_pos[:, None, :] + ro  # (K, n_fingers, 3)
        e_H = jnp.linalg.norm(fingertip_pos - contact_world, axis=-1)  # (K, n_fingers)

        # e_O: distance from contact point to cylinder surface (in object frame)
        # Cylinder parameters (should be passed via info or p; hardcoded here)
        R = 0.05  # radius
        H = 0.1  # half-height
        xy = ro[..., :2]  # (K, n_fingers, 2)
        z = ro[..., 2]  # (K, n_fingers)
        radial = jnp.linalg.norm(xy, axis=-1)  # (K, n_fingers)

        # Distance to side: |radial - R|
        d_side = jnp.abs(radial - R)
        # Distance to cap: if |z| > H, distance to nearest cap
        d_cap = jnp.where(jnp.abs(z) > H, jnp.abs(jnp.abs(z) - H), 0.0)
        # Total e_O (approximation)
        e_O = d_side + d_cap

        ci_cost = jnp.sum(e_H ** 2 + e_O ** 2)

        # ------------------------------------------------------------------
        # Physics cost (L_physics)
        # ------------------------------------------------------------------
        dt_phase = p.delT_phase
        # Object acceleration via finite differences
        obj_acc = (obj_vel[1:] - obj_vel[:-1]) / dt_phase  # (K-1,3)
        # Sum of forces at each phase (excluding last because acceleration undefined)
        F_total = jnp.sum(forces, axis=1)  # (K,3)
        newton_error = jnp.mean(jnp.sum((F_total[1:] - p.mass * obj_acc) ** 2, axis=1))

        # Force regularization
        force_reg = p.lamb * jnp.mean(jnp.sum(forces ** 2, axis=(1, 2)))

        # Friction cone (simplified: assume all contacts on cylinder side)
        # Normals are radial directions in the xy-plane
        xy_norm = jnp.linalg.norm(ro[..., :2], axis=-1, keepdims=True) + 1e-8
        n_radial = ro[..., :2] / xy_norm  # (K, n_fingers, 2)
        n = jnp.concatenate([n_radial, jnp.zeros_like(ro[..., 2:3])], axis=-1)  # (K, n_fingers, 3)

        f_normal = jnp.sum(forces * n, axis=-1)  # (K, n_fingers)
        f_tangent = jnp.linalg.norm(forces - f_normal[..., None] * n, axis=-1)
        cone_cost = jnp.mean(jnp.sum(jnp.maximum(f_tangent - p.mu * f_normal, 0.0) ** 2, axis=1))

        # ------------------------------------------------------------------
        # Task cost (L_task)
        # ------------------------------------------------------------------
        goal_pos = jnp.array([goals[0].x, goals[0].y, goals[0].z])
        task_cost = jnp.sum((obj_pos[-1] - goal_pos) ** 2)

        # ------------------------------------------------------------------
        # Total cost (all weights = 1 for simplicity)
        # ------------------------------------------------------------------
        total = ci_cost + newton_error + force_reg + cone_cost + task_cost
        return total

    # ----------------------------------------------------------------------
    # Optimization with JAXopt L-BFGS
    # ----------------------------------------------------------------------
    def optimize_cio(self, S_init, p, goals, info):
        # Define loss (only S varies)
        def loss(S):
            return self.cost_function(S, p, goals, info)

        # JIT compile loss and gradient
        # loss_jit = jit(loss)
        # grad_loss = jit(grad(loss))

        # Set up L-BFGS solver
        solver = LBFGS(fun=loss, maxiter=100, tol=1e-4, history_size=10)

        # Run
        result = solver.run(S_init)
        return result.params


# ----------------------------------------------------------------------
# Visualization (using standard MuJoCo)
# ----------------------------------------------------------------------
def visualize_trajectory(info, p, S_opt, goals, outfile='traj.gif'):
    """Replay optimized trajectory in MuJoCo and create a GIF."""
    per_phase = p.per_phase
    n_joints = p.n_joints
    phases = S_opt.reshape(p.K, per_phase)

    obj_pos_ph = phases[:, 0:3]  # (K,3)
    joint_angles_ph = phases[:, 6:6 + n_joints]  # (K, n_joints)

    # Interpolate to full time grid (linear)
    t_ph = np.linspace(0, p.T_final, p.K)
    t_full = np.linspace(0, p.T_final, p.T_steps + 1)
    obj_pos_full = np.zeros((p.T_steps + 1, 3))
    joint_full = np.zeros((p.T_steps + 1, n_joints))
    for i in range(3):
        obj_pos_full[:, i] = np.interp(t_full, t_ph, obj_pos_ph[:, i])
    for i in range(n_joints):
        joint_full[:, i] = np.interp(t_full, t_ph, joint_angles_ph[:, i])

    # MuJoCo model and data
    m = info['mj_model']
    d = info['mj_data']

    # Set initial object pose (from first full time step)
    obj_adr = info['obj_qposadr']
    d.qpos[obj_adr:obj_adr + 3] = obj_pos_full[0]
    d.qpos[obj_adr + 3:obj_adr + 7] = [1, 0, 0, 0]  # identity quat

    # Render video
    start_render = time.time()
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.distance = 1
    cam.azimuth = 135
    cam.elevation = 2
    cam.lookat = [.2, -.2, 0.5]

    world.model.vis.global_.fovy = 60
    # Visual options
    vopt = mj.MjvOption()
    # vopt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
    pert = mj.MjvPerturb()  # Empty MjvPerturb object
    catmask = mj.mjtCatBit.mjCAT_DYNAMIC

    # Create off-screen renderer
    assert world.mj_recorder.start()
    for t_idx in range(p.T_steps + 1):
        # Set hand joints
        for jnt_idx, adr in enumerate(info['joint_qposadr']):
            d.qpos[adr] = joint_full[t_idx, jnt_idx]
        # Set object pose
        d.qpos[obj_adr:obj_adr + 3] = obj_pos_full[t_idx]
        mj.mj_step(m, d)

        # Use first model to make the scene, add subsequent models
        if t_idx == 0:
            world.mj_renderer.update_scene(d, cam, scene_option=vopt)
        else:
            mj.mjv_addGeoms(m, d, vopt, pert, catmask, world.mj_renderer.scene)

        # Render and add the frame
        world.mj_recorder.add_frame(world.mj_renderer.render())

    world.mj_recorder.stop()
    end_render = time.time()
    print(f'Rendering time {end_render - start_render:.1f} seconds')


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Load model and prepare info
    world = MujocoWorld(record_video=False)
    info = world.info

    # Set CIO parameters
    p = Params(
        n_fingers=info['n_fingers'],
        n_joints=info['n_joints'],
        K=8,
        delT=0.02,
        delT_phase=0.25,
        mass=0.05,
        mu=0.5,
        lamb=1e-3
    )

    # Goal (move object to (0.1, 0.0, 0.2))
    goals = [Position(0.1, 0.0, 0.2)]

    # Initial object pose and velocity
    init_obj_pos = world.data.body(world.object_id).xpos.copy()
    init_obj_vel = np.array([0.0, 0.0, 0.0])

    # Create initial guess
    S_init = world.init_trajectory(info, p, init_obj_pos, init_obj_vel)

    # Run optimization
    print("Starting optimization...")
    S_opt = world.optimize_cio(S_init, p, goals, info)
    print("Optimization finished.")

    # Visualize result
    # Visualize result
    with Xvfb(width=640, height=480) as xvfb:
        print(f"Using Xvfb display: {xvfb.new_display}")
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        assert os.environ["DISPLAY"] is not None, "Xvfb is required to start in advance!\n"
        "Please use xvfbwrapper. DON'T RUN: `Xvfb :<no> -screen 0 720x480x24` DIRECTLY!"
        visualize_trajectory(info, p, S_opt, goals)
