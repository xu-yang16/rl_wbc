from isaacgym import gymapi, gymutil


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    sim = gym.create_sim(
        sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params
    )

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    else:
        viewer = None
    return gym, sim, viewer
