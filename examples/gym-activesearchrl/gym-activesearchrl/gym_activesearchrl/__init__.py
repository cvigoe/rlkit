from gym.envs.registration import register

register(
    id='activesearchrl-v0',
    entry_point='gym_activesearchrl.envs:ActiveSearchRL',
)