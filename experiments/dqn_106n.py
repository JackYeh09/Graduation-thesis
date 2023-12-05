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
    env1 = SumoEnvironment(
        net_file="/workspace/pj/sumo-rl/nets/n001/n002.net.xml",
        route_file="/workspace/pj/sumo-rl/nets/n001/n002.rou.xml",
        out_csv_name="/workspace/pj/sumo-rl/outputs/n001/dqn",
        
        single_agent=True,
        use_gui=True,
        num_seconds=14400,
        reward_fn="pressure",
    )
    
    env1.reset()
    
    count=0
    
    while(True):
        env1.sumo.person.add(
            personID=f"ped{count}",
            edgeID="-E1",
            pos=10.0,
            )
        env1.sumo.person.appendWalkingStage(
            personID=f"ped{count}",
            edges=["-E0","E1"],
            arrivalPos=10,
            )
        env1.step(0)
        
        sum = 0
        
        for i in env1.sumo.edge.getIDList():
            sum+=len(env1.sumo.edge.getLastStepPersonIDs(i))
        print(sum)

        count+=1


        # model = DQN(
        #     env=env1,
        #     policy="MlpPolicy",
        #     learning_rate=0.001,
        #     learning_starts=0,
        #     train_freq=1,
        #     target_update_interval=500,
        #     exploration_initial_eps=0.05,
        #     exploration_final_eps=0.01,
        #     verbose=1,
        # )
        # model.learn(total_timesteps=14400)
