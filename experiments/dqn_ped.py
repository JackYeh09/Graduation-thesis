import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3 import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="/workspace/pj/sumo-rl/nets/n001/n002.net.xml",
        route_file="/workspace/pj/sumo-rl/nets/n001/n002.rou.xml",
        out_csv_name="/workspace/pj/sumo-rl/outputs/n001/dqn",
        single_agent=True,
        use_gui=True,
        num_seconds=14400,
        reward_fn="pressure",
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=14400)

    model.save("ped_weight", ":/")