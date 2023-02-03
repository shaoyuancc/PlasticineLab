from plb.envs import  make
import sys

env_name = "Move-v5"
episode_length = 20

class Agent():
    def __init__(self, env):
        pass
        
    def get_action(self, state):        
        action = [1] * 6
        return action

env = make(env_name)
agent = Agent(env)
rewards = list()

for i_episode in range(1):
    observation = env.reset()
    for t in range(episode_length):
        print(f"step {t}")
        env.render()
        action = agent.get_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            rewards.append(reward)
            
            break
            
print(f"Final Rewards:{rewards}")
env.close()