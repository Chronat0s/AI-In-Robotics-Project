from gym.envs.registration import register
register(
    id='cobot_ai4robotics',
    entry_point='cobot_ai4robotics.envs:CobotAI4RoboticsEnv'
)