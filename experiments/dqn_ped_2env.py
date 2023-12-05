#%%
import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3 import DQN

#%%

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

#%%

# [[route_file, out_csv_name]]
files = [
    ["/workspace/pj/sumo-rl/nets/n001/n002.rou.xml", "/workspace/pj/sumo-rl/outputs/n002/dqn"],
    ["/workspace/pj/sumo-rl/nets/n001/n003.rou.xml", "/workspace/pj/sumo-rl/outputs/n003/dqn"],
]
envs = []


#%%

flag = True
for f in files:
    env = SumoEnvironment(
        net_file="/workspace/pj/sumo-rl/nets/n001/n002.net.xml",
        route_file = f[0],
        out_csv_name = f[1],
        single_agent = True,
        use_gui = True,
        num_seconds = 14400,
        reward_fn = "pressure",
    )
    
    if flag:
        model = DQN(
            env = env,
            policy="MlpPolicy",
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=1,
        ) 
        flag = False
    else:
        model.set_env(env)
    
    model.learn(total_timesteps=14400)
    
    env.reset()
    
    # print(env.metrics)
    


# %%

model.save("ped2env_weight", ":/")
