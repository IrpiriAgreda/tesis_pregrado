from gym.envs.registration import register

register(
    id='city_simulation-v0',
    entry_point='city_simulation.envs:CitySimulation',
)
