from gym.envs.registration import register
try:
    from .pointmaze import U_MAZE
except ImportError:
    # d4rl is not installed everywhere; only the point_maze envs need it, and registration is
    # lazy (the entry point is imported by gym.make, not here), so leaving maze_spec=None keeps
    # every other env in this file registerable and fails only if point_maze is actually used.
    U_MAZE = None
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id='point_maze',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# ManiSkill tasks. max_episode_steps matches ManiSkill's own registration for each task (and the
# recorded trajectory length). The env checker is disabled because, like every wrapper here,
# reset() returns (obs, state) rather than the (obs, info) gym expects.
register(
    id="push_cube",
    entry_point="env.pushcube.pushcube_wrapper:PushCubeWrapper",
    max_episode_steps=50,
    disable_env_checker=True,
)

register(
    id="lift_peg",
    entry_point="env.liftpeg.liftpeg_wrapper:LiftPegUprightWrapper",
    max_episode_steps=50,
    disable_env_checker=True,
)

register(
    id="place_sphere",
    entry_point="env.placesphere.placesphere_wrapper:PlaceSphereWrapper",
    max_episode_steps=50,
    disable_env_checker=True,
)
