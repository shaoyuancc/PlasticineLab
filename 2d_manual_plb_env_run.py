from plb.envs import  make
import numpy as np

env_name = "2DPinch-v1"
episode_length = 30

class Agent():
    def __init__(self, env):
        self.mode = 0
        pass
        
    def get_action(self, state):
        manipulator_state = state[-7:]
        print(manipulator_state) # [xpos, ypos, zpos, 4 rotation states]
        # if self.mode == 0: # Descend
        #     action = [0, -1]
        #     if manipulator_state[1] <= 0.15:
        #         self.mode = 1
        
        # if self.mode == 1: # Ascend
        #     action = [0, 1]
        #     if manipulator_state[1] >= 0.35:
        #         self.mode = 2
        
        # if self.mode == 2: # Stay still
        #     action = [0, 0]
        action = [0,0,0]
        return action

env = make(env_name)
agent = Agent(env)
rewards = list()

for i_episode in range(1):
    observation = env.reset()
    for t in range(episode_length):
        print(f"step {t}")
        # env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            rewards.append(reward)
            break
            
print(f"Final Rewards:{rewards}")
env.close()