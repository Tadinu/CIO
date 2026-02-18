"""
CIO for Allegro Hand + Cylinder in MuJoCo (3D)
Requires: mujoco>=3.0, numpy, scipy, imageio, matplotlib
Place an Allegro hand MJCF model (e.g., 'allegro_hand_right.xml') and adjust paths.
"""

import time
from xvfbwrapper import Xvfb
import numpy as np
import mujoco as mj
import mujoco.viewer
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import os
from collections import namedtuple

from mjmanip.robot.panda_leap import PandaLeapEnv, PandaLeap, ARM_SCENE_XML_PATH, ARM_XML_PATH, HAND_XML_PATH
from video_recorder import VideoRecorder

# ------------------------------------------------------------------------------
# Data structures (extended to 3D)
# ------------------------------------------------------------------------------
Position = namedtuple('Position', 'x y z')
Pose = namedtuple('Pose', 'x y z quat')  # quat = [w, x, y, z]
LinearVelocity = namedtuple('LinearVelocity', 'x y z')
Velocity = namedtuple('Velocity', 'x y z wx wy wz')
Acceleration = namedtuple('Acceleration', 'x y z wx wy wz')
Contact = namedtuple('Contact', 'f ro c')  # f: force (3), ro: contact pos in obj frame (3), c: probability


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
        self.robot_env = PandaLeapEnv(world_scene_xml=ARM_SCENE_XML_PATH,
                                      arm_xml=ARM_XML_PATH,
                                      hand_xml=HAND_XML_PATH)
        spec = self.robot_env.construct_main_spec(self.robot_env.meshdir, self.robot_env.texturedir)
        spec.option.timestep = 0.01
        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        # Cache body ids
        self.fingertip_ids = [self.model.body(name).id
                              for name in PandaLeap.hand_items_full_names(PandaLeap.FINGER_TIPS_BODY_NAMES)]
        self.object_id = self.model.body(PandaLeap.OBJECT_NAMES[0]).id

        # Number of fingers (contacts)
        self.n_fingers = len(self.fingertip_ids)

        # Identify joint indices (assume all joints are actuated and used as variables)
        self.joint_names = []
        self.joint_dofs = []  # starting index in qpos for each joint
        self.joint_qposadr = []
        self.joint_actadr = []
        self.joint_limits = []
        for jnt_id in range(self.model.njnt):
            jnt = self.model.joint(jnt_id)
            if jnt.type == mj.mjtJoint.mjJNT_FREE:  # skip object's free joint
                continue
            self.joint_names.append(jnt.name)
            self.joint_qposadr.append(jnt.qposadr[0])
            self.joint_dofs.append(jnt.dofadr[0])  # for velocity indexing
            self.joint_limits.append((jnt.range[0], jnt.range[1]))
        self.n_joints = len(self.joint_qposadr)

        self.joint_actadr = list(range(self.model.nu))

        # Initial configuration (qpos0)
        self.qpos0 = self.data.qpos.copy()
        self.qvel0 = self.data.qvel.copy()

        # For storing per-timestep contact data (filled during trajectory generation)
        self.contact_data = None  # list of dicts per timestep

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

    def set_joint_angles(self, joint_angles, kinematics_mode=False):
        """Set hand joint angles (in radians)."""
        if kinematics_mode:
            for i, adr in enumerate(self.joint_qposadr):
                self.data.qpos[adr] = joint_angles[i]
        else:
            for i, adr in enumerate(self.joint_actadr):
                self.data.ctrl[adr] = joint_angles[i]

    def get_fingertip_pose(self, finger_idx):
        """Return Pose (position + quat) of fingertip body."""
        body_id = self.fingertip_ids[finger_idx]
        pos = self.data.body(body_id).xpos.copy()
        quat = self.data.body(body_id).xquat.copy()  # [w,x,y,z]
        return Pose(pos[0], pos[1], pos[2], quat)

    def get_fingertip_velocity(self, finger_idx):
        """Return spatial velocity (linear + angular) of fingertip body."""
        body_id = self.fingertip_ids[finger_idx]
        # MuJoCo provides body linear and angular velocity in the world frame
        lin = self.data.body(body_id).cvel[
            :3]  # correct? cvel is spatial velocity in world? Actually cvel is in body frame? Let's use xpos derivative.
        # Simpler: use finite differences during trajectory generation, not here.
        # We'll compute velocities via finite differences later.
        return Velocity(0, 0, 0, 0, 0, 0)  # placeholder

    def get_object_pose(self):
        body_id = self.object_id
        pos = self.data.body(body_id).xpos.copy()
        quat = self.data.body(body_id).xquat.copy()
        return Pose(pos[0], pos[1], pos[2], quat)

    def get_object_velocity(self):
        body_id = self.object_id
        # Use mj_objectVelocity to get 6D velocity? For simplicity we'll use finite differences later.
        return Velocity(0, 0, 0, 0, 0, 0)

    def get_contact_distance_and_normal(self, finger_idx, point_on_object):
        """
        Compute signed distance from fingertip sphere to object surface at given object point.
        point_on_object: 3D point in world coordinates.
        Returns distance (positive if outside) and normal vector (pointing from object to fingertip).
        For a cylinder, we need the closest point on cylinder surface to the fingertip position.
        This is a simplified geometric check (object is assumed to be a cylinder with known radius and height).
        """
        # For now, assume object is a cylinder with radius R and half-height H.
        # Get fingertip position
        ft_pos = self.get_fingertip_pose(finger_idx)[:3]
        # Compute closest point on cylinder (infinite cylinder approximation)
        # Project onto cylinder axis (z-axis of object)
        obj_pose = self.get_object_pose()
        obj_pos = obj_pose[:3]
        obj_quat = obj_pose[3:]
        # Convert ft_pos to object frame
        # For simplicity, assume object is axis-aligned (no rotation). Extend later.
        # Placeholder: compute radial distance
        R = 0.05  # cylinder radius, should be stored in object
        axis = np.array([0, 0, 1])  # cylinder axis
        # Vector from object center to fingertip
        d = ft_pos - obj_pos
        # Component along axis
        h = np.dot(d, axis)
        # Clamp to half-height
        H = 0.1  # cylinder half-height
        if abs(h) > H:
            # Cap region
            closest_ax = axis * np.sign(h) * H
            closest_rad = d - h * axis
        else:
            # Side region
            closest_ax = h * axis
            # Radial direction
            radial = d - h * axis
            radial_norm = np.linalg.norm(radial)
            if radial_norm > 1e-8:
                closest_rad = radial * (R / radial_norm)
            else:
                closest_rad = np.zeros(3)
        closest_point = obj_pos + closest_ax + closest_rad
        # Distance (signed: positive if outside)
        dist = np.linalg.norm(ft_pos - closest_point) - 0.01  # fingertip radius approx
        # Normal from object to fingertip
        n = (ft_pos - closest_point)
        if np.linalg.norm(n) > 1e-8:
            n = n / np.linalg.norm(n)
        else:
            n = np.zeros(3)
        return dist, n

    def check_self_collision(self):
        """Check collisions between fingers (simplified)."""
        # Use MuJoCo's collision detection? For now return 0.
        return 0.0


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
class Params:
    def __init__(self, world, K=10, delT=0.05, delT_phase=0.5, mass=0.1, mu=0.5, lamb=1e-3):
        self.K = K
        self.delT = delT
        self.delT_phase = delT_phase
        self.mass = mass
        self.mu = mu
        self.lamb = lamb
        self.N = world.n_fingers
        self.steps_per_phase = int(self.delT_phase / self.delT)
        self.T_steps = self.K * self.steps_per_phase
        self.T_final = self.K * self.delT_phase

        # Variable dimensions per phase:
        #   joint angles: n_joints
        #   contact forces: N * 3
        #   contact positions (in object frame): N * 3
        self.n_vars_per_phase = world.n_joints + 6 * world.n_fingers
        self.len_s = self.n_vars_per_phase
        self.len_S = self.len_s * self.K


# ------------------------------------------------------------------------------
# Trajectory generation and interpolation
# ------------------------------------------------------------------------------
def generate_initial_trajectory(world, goals, p):
    """
    Create a naive initial guess: hand starts in qpos0, object moves linearly to goal.
    Variables: [joint_angles_phase1, f_phase1, ro_phase1, ...]
    """
    S = np.zeros(p.len_S)
    # Joint angles: keep initial pose constant (or interpolate to a pre-grasp)
    q0 = np.array([world.data.qpos[adr] for adr in world.joint_qposadr])
    for k in range(p.K):
        idx = k * p.len_s
        S[idx: idx + world.n_joints] = q0  # constant joints (will be optimized)
        # Forces: small random
        S[idx + world.n_joints: idx + world.n_joints + 3 * world.n_fingers] = 0.01 * np.random.randn(
            3 * world.n_fingers)
        # Contact positions: place near object surface (e.g., on object center)
        obj_pose = world.get_object_pose()
        obj_center = np.array([obj_pose.x, obj_pose.y, obj_pose.z])
        ro = np.tile(obj_center, world.n_fingers)
        S[idx + world.n_joints + 3 * world.n_fingers: idx + world.n_joints + 6 * world.n_fingers] = ro
    return S


def interpolate_trajectory(S, world, p):
    """
    Convert the phase-level variables S into a full time series (T_steps+1).
    Returns a list of world states (with dynamics filled) for each time step.
    """
    # Reshape S into phases
    phases = S.reshape((p.K, p.len_s))
    # For each time step, we need joint angles, forces, contact positions.
    # We'll use cubic splines for joint angles and linear for forces/ro.
    # Prepare arrays
    times_phase = np.linspace(0, p.T_final, p.K + 1)  # include initial state at t=0
    joint_traj = np.zeros((world.n_joints, p.K + 1))
    force_traj = np.zeros((world.n_fingers, 3, p.K + 1))
    ro_traj = np.zeros((world.n_fingers, 3, p.K + 1))

    # Initial state (phase 0) from world.s0? We need an initial world state.
    # For simplicity, we'll assume the initial world state corresponds to the first column.
    # Use initial joint angles from world (qpos0) for t=0.
    q0 = np.array([world.data.qpos[adr] for adr in world.joint_qposadr])
    joint_traj[:, 0] = q0
    force_traj[:, :, 0] = 0.0
    ro_traj[:, :, 0] = 0.0  # placeholder, should be consistent with initial contacts

    for k in range(p.K):
        vars_k = phases[k]
        joint_traj[:, k + 1] = vars_k[:world.n_joints]
        f_flat = vars_k[world.n_joints: world.n_joints + 3 * world.n_fingers].reshape((world.n_fingers, 3))
        ro_flat = vars_k[world.n_joints + 3 * world.n_fingers:].reshape((world.n_fingers, 3))
        force_traj[:, :, k + 1] = f_flat
        ro_traj[:, :, k + 1] = ro_flat

    # Spline interpolation for joints
    joint_splines = []
    for i in range(world.n_joints):
        spl = CubicSpline(times_phase, joint_traj[i, :], bc_type='natural')
        joint_splines.append(spl)

    # Linear interpolation for forces and contact positions (simple)
    times_full = np.linspace(0, p.T_final, p.T_steps + 1)
    world_states = []
    for t_idx, t in enumerate(times_full):
        # Find phase index
        phase_idx = int(t // p.delT_phase)
        if phase_idx >= p.K:
            phase_idx = p.K - 1
        alpha = (t - phase_idx * p.delT_phase) / p.delT_phase
        # Joint angles from spline
        joint_angles = np.array([joint_splines[i](t) for i in range(world.n_joints)])
        world.set_joint_angles(joint_angles)
        mj.mj_step(world.model, world.data)

        # Forces and contact positions: linear blend
        if phase_idx < p.K - 1:
            f = (1 - alpha) * force_traj[:, :, phase_idx] + alpha * force_traj[:, :, phase_idx + 1]
            ro = (1 - alpha) * ro_traj[:, :, phase_idx] + alpha * ro_traj[:, :, phase_idx + 1]
        else:
            f = force_traj[:, :, phase_idx]
            ro = ro_traj[:, :, phase_idx]

        # Build a world snapshot (similar to original World object) with current state
        snapshot = {
            'time': t,
            'joint_angles': joint_angles,
            'fingertip_poses': [world.get_fingertip_pose(i) for i in range(world.n_fingers)],
            'object_pose': world.get_object_pose(),
            'contact_forces': f,
            'contact_positions_obj': ro,
            'contact_prob': np.ones(world.n_fingers)  # assume always in contact
        }
        world_states.append(snapshot)

    return world_states


# ------------------------------------------------------------------------------
# Cost functions (adapted to 3D)
# ------------------------------------------------------------------------------
def compute_e_H_and_e_O(world, snapshot, finger_idx):
    """
    Compute the two contact invariants:
      e_H: distance from contact point to fingertip surface
      e_O: distance from contact point to object surface
    """
    ft_pose = snapshot['fingertip_poses'][finger_idx]
    ft_pos = np.array([ft_pose.x, ft_pose.y, ft_pose.z])
    ro = snapshot['contact_positions_obj'][finger_idx]  # in object frame
    obj_pose = snapshot['object_pose']
    obj_pos = np.array([obj_pose.x, obj_pose.y, obj_pose.z])
    obj_quat = obj_pose.quat
    # Transform ro to world frame using object orientation
    # For simplicity, assume object is axis-aligned (skip rotation). In general use quaternion.
    r_world = obj_pos + ro  # simplified

    # Distance from r_world to fingertip (sphere radius approx)
    d_ft = np.linalg.norm(r_world - ft_pos) - 0.01  # fingertip radius
    e_H = max(d_ft, 0.0)  # positive if outside

    # Distance from r_world to object surface (cylinder)
    # We need signed distance to cylinder. For now, use the same function as before.
    # Actually we already computed the closest point on object to r_world? We want distance from r_world to surface.
    # Since ro is supposed to be on object surface, we should penalize deviation.
    # For a cylinder, we can compute the distance from ro (in object frame) to the cylinder surface.
    # Assume cylinder radius R and half-height H.
    R = 0.05;
    H = 0.1
    # In object frame, ro is a point. Check if it lies on cylinder surface.
    # Radial distance from axis (z)
    r_radial = np.sqrt(ro[0] ** 2 + ro[1] ** 2)
    z = ro[2]
    if abs(z) <= H:
        # on side: expected r_radial = R
        d_surface = abs(r_radial - R)
    else:
        # on cap: expected distance from cap center?
        # For simplicity, compute distance to nearest point on cylinder surface.
        # Not implemented fully. We'll use a placeholder.
        d_surface = 0.0
    e_O = d_surface

    return e_H, e_O


def L_CI(snapshot, world, p):
    cost = 0.0
    for fi in range(world.n_fingers):
        e_H, e_O = compute_e_H_and_e_O(world, snapshot, fi)
        # Time derivatives not computed here; could be added via finite diff later
        cost += e_H ** 2 + e_O ** 2
    return cost


def L_physics(snapshot, prev_snapshot, next_snapshot, world, p):
    """
    Physics cost: Newton's 2nd law, force cone, force regularization.
    Use finite differences for acceleration.
    """
    if prev_snapshot is None or next_snapshot is None:
        return 0.0
    # Object acceleration from finite difference
    dt = p.delT
    obj_pos_prev = np.array(
        [prev_snapshot['object_pose'].x, prev_snapshot['object_pose'].y, prev_snapshot['object_pose'].z])
    obj_pos_curr = np.array([snapshot['object_pose'].x, snapshot['object_pose'].y, snapshot['object_pose'].z])
    obj_pos_next = np.array(
        [next_snapshot['object_pose'].x, next_snapshot['object_pose'].y, next_snapshot['object_pose'].z])
    vel_curr = (obj_pos_next - obj_pos_prev) / (2 * dt)  # central difference
    accel = (obj_pos_next - 2 * obj_pos_curr + obj_pos_prev) / (dt ** 2)

    # Sum of forces (weighted by probability)
    F_total = np.zeros(3)
    for fi in range(world.n_fingers):
        F_total += snapshot['contact_forces'][fi]  # assume probability = 1
    newton_error = np.linalg.norm(F_total - p.mass * accel) ** 2

    # Force regularization
    force_reg = p.lamb * np.sum([np.linalg.norm(f) ** 2 for f in snapshot['contact_forces']])

    # Friction cone (simplified: force within cone)
    cone_cost = 0.0
    for fi in range(world.n_fingers):
        f = snapshot['contact_forces'][fi]
        # Get normal at contact point (need surface normal at ro)
        # We'll compute using cylinder geometry in object frame
        ro = snapshot['contact_positions_obj'][fi]
        # For side: normal is radial direction
        r_norm = np.linalg.norm(ro[:2])
        if r_norm > 1e-6:
            n = np.array([ro[0] / r_norm, ro[1] / r_norm, 0])
        else:
            n = np.array([0, 0, 1])  # approximate for cap
        # Transform normal to world frame (simplified: assume object axis-aligned)
        f_normal = np.dot(f, n)
        f_tangent = np.linalg.norm(f - f_normal * n)
        cone_cost += max(f_tangent - p.mu * f_normal, 0.0) ** 2

    return newton_error + force_reg + cone_cost


def L_task(snapshot, goals, p):
    """Goal: object position at final time."""
    # For simplicity, only final time considered (I=1 at last step)
    # In full version, we would check time index.
    cost = 0.0
    obj_pos = np.array([snapshot['object_pose'].x, snapshot['object_pose'].y, snapshot['object_pose'].z])
    for goal in goals:
        if isinstance(goal, Position):
            goal_pos = np.array([goal.x, goal.y, goal.z])
            cost += np.linalg.norm(obj_pos - goal_pos) ** 2
    return cost


def total_cost(S, goals, world, p, stage=0):
    """
    Wrapper for scipy.optimize.
    S : flat array of all variables (K * n_vars_per_phase)
    """
    # Generate full trajectory
    world_states = interpolate_trajectory(S, world, p)

    # Precompute time derivatives for e_dot if needed (skip for now)

    total = 0.0
    # Stage weights (simplified: use constant)
    w_CI = 1.0
    w_physics = 1.0
    w_task = 1.0

    # We'll accumulate costs over time
    for t_idx, snapshot in enumerate(world_states):
        # CI cost
        ci_cost = w_CI * L_CI(snapshot, world, p)
        # Physics cost (needs neighbors)
        prev = world_states[t_idx - 1] if t_idx > 0 else None
        nxt = world_states[t_idx + 1] if t_idx < len(world_states) - 1 else None
        phys_cost = w_physics * L_physics(snapshot, prev, nxt, world, p)
        # Task cost (only at final step)
        if t_idx == len(world_states) - 1:
            task_cost = w_task * L_task(snapshot, goals, p)
        else:
            task_cost = 0.0
        total += ci_cost + phys_cost + task_cost

    return total


# ------------------------------------------------------------------------------
# Main optimization routine
# ------------------------------------------------------------------------------
def CIO(goals, world, p):
    # Initial guess
    S0 = generate_initial_trajectory(world, goals, p)

    # Bounds: joint limits from model, force bounds (e.g., +/-10N), contact positions near object
    bounds = []
    for k in range(p.K):
        # Joint angles (per phase)
        for jnt_idx in range(world.n_joints):
            low, high = world.joint_limits[jnt_idx]
            bounds.append((low, high))
        # Forces (per finger, per coordinate)
        for _ in range(world.n_fingers):
            for _ in range(3):
                bounds.append((-10.0, 10.0))  # force limits (N)
        # Contact positions (per finger, per coordinate)
        for _ in range(world.n_fingers):
            for _ in range(3):
                bounds.append((-0.2, 0.2))  # relative to object center (m)

    # Sanity check
    assert len(bounds) == len(S0), f"Bounds length {len(bounds)} != S0 length {len(S0)}"

    # Run L-BFGS-B
    res = minimize(fun=total_cost, x0=S0, args=(goals, world, p, 0),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 100, 'ftol': 1e-4, 'eps': 1e-3})

    print("Optimization finished. Cost:", res.fun)
    return res.x


# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------
def visualize_trajectory(world, p, S_opt, output_video: bool = True):
    """Generate an animation of the optimized trajectory."""
    world_states = interpolate_trajectory(S_opt, world, p)

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

    # Simulate and render
    model = world.model
    data = world.data
    if output_video:
        assert world.mj_recorder.start()
    for t_idx, snapshot in enumerate(world_states):
        # Set joint angles and step simulation
        world.set_joint_angles(snapshot['joint_angles'])
        mj.mj_step(model, data)
        # Render with MuJoCo's offscreen context
        print(f"Rendering frame {t_idx}")

        # Use first model to make the scene, add subsequent models
        if t_idx == 0:
            world.mj_renderer.update_scene(data, cam, scene_option=vopt)
        else:
            mj.mjv_addGeoms(model, data, vopt, pert, catmask, world.mj_renderer.scene)

        # Render and add the frame
        pixels = world.mj_renderer.render()
        if output_video:
            world.mj_recorder.add_frame(pixels)

    # In a full implementation, you would save frames and create GIF.
    print("Visualization not fully implemented; see comments for MuJoCo rendering.")

    if world.mj_recorder:
        world.mj_recorder.stop()

    end_render = time.time()
    print(f'Rendering time {end_render - start_render:.1f} seconds')


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Load model (adjust paths)
    world = MujocoWorld()

    mj.mj_step(world.model, world.data)

    # Goals: move object to (0.1, 0.0, 0.2)
    goals = [Position(0.1, 0.0, 0.2)]

    # Parameters
    p = Params(world, K=200, delT=0.01, delT_phase=0.1, mass=0.05, mu=0.5)

    # Run optimization
    S_opt = CIO(goals, world, p)

    # Visualize result
    with Xvfb(width=640, height=480) as xvfb:
        print(f"Using Xvfb display: {xvfb.new_display}")
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        assert os.environ["DISPLAY"] is not None, "Xvfb is required to start in advance!\n"
        "Please use xvfbwrapper. DON'T RUN: `Xvfb :<no> -screen 0 720x480x24` DIRECTLY!"
        visualize_trajectory(world, p, S_opt)
