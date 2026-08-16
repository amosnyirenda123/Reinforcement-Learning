import gymnasium

gymnasium.register(
    id="BanditTwoArmedDeterministicFixed-v0",
    entry_point="envs.bandits.bandit:BanditTwoArmedDeterministicFixed",
)

gymnasium.register(
    id="BanditTwoArmedHighHighFixed-v0",
    entry_point="envs.bandits.bandit:BanditTwoArmedHighHighFixed",
)

gymnasium.register(
    id="BanditTwoArmedLowLowFixed-v0",
    entry_point="envs.bandits.bandit:BanditTwoArmedLowLowFixed",
)

gymnasium.register(
    id="BanditTwoArmedHighLowFixed-v0",
    entry_point="envs.bandits.bandit:BanditTwoArmedHighLowFixed",
)