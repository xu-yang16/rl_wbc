"""Default simulation config."""

from isaacgym import gymapi
from ml_collections import ConfigDict


def get_asset_config():
    config = ConfigDict()
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = 3
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01
    asset_options.disable_gravity = False
    config.asset_options = asset_options
    config.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
    return config


def get_sim_config(
    use_gpu=True, show_gui=True, sim_dt=0.002, use_penetrating_contact=True
):
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.dt = sim_dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UpAxis(gymapi.UP_AXIS_Z)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.num_subscenes = 0  # default_args.subscenes
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1  # 0: pgs, 1: tgs
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    if use_penetrating_contact:
        sim_params.physx.contact_offset = 0.0
        sim_params.physx.rest_offset = -0.01
    else:
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.5  # 0.5 [m/s]
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 2**23  # 2**24 needed for 8000+ envs
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.ContactCollection(
        2
    )  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    config = ConfigDict()
    config.sim_device = "cuda" if use_gpu else "cpu"
    config.show_gui = show_gui
    config.physics_engine = gymapi.SIM_PHYSX
    config.sim_params = sim_params
    config.action_repeat = 1
    config.dt = sim_dt
    return config
