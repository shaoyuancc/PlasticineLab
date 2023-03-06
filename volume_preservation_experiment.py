from plb.envs import  make
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# EXPERIMENT CONFIGURATION
should_render = False
save_output = True

env_names = [f"BallSquish-v{i}" for i in range(1,2)] # BallSquish, BallPlaten
episode_length = 250
action_magnitude = 0.05
action_lowest_height = 0.15

class Agent():
    def __init__(self, env):
        self.mode = 0
        pass
        
    def get_action(self, state):
        manipulator_state = state[-9:]
        print(manipulator_state) # [xpos, ypos, zpos, 4 rotation states, vol_particles, vol_grid]
        if self.mode == 0: # Descend
            action = [0, -action_magnitude, 0]
            if manipulator_state[1] <= action_lowest_height:
                self.mode = 1
        
        if self.mode == 1: # Ascend
            action = [0, action_magnitude, 0]
            if manipulator_state[1] >= 0.35:
                self.mode = 2
        
        if self.mode == 2: # Stay still
            action = [0, 0, 0]

        return action

variant_names = list()
final_positions = list()

print(f"ENV NAMES: {env_names}")
columns= ['step', 'manipulator position', 'vol (particles)','vol (grid)', 'variant']
df = pd.DataFrame(columns=columns)

for env_name in env_names:
    print(f"Running {env_name}")
    env = make(env_name, include_vol_in_state=True)
    agent = Agent(env)
    rewards = list()

    variant_names.append(f"E={env.cfg_sim.E:.0e}, nu={env.cfg_sim.nu:.0e}")
    
    observation = env.reset()
    for t in range(episode_length):
        print(f"step {t}")
        if should_render:
            env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        only_volumes = observation[-2:]
        #      ['step', 'manipulator position', 'vol (particles)','vol (grid)', 'variant']
        vals = [t, observation[-8], observation[-2], observation[-1], variant_names[-1]]
        new_row = dict(zip(columns, vals))
        df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    n = env.cfg.n_observed_particles
    
    last_cfg_env = env.cfg
    last_cfg_sim = env.cfg_sim
    env.close()


def create_interactive_plot(df: pd.DataFrame, annotation: str, save_output: bool=False, base_filename: str=""):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for variant in variant_names:
        filtered_df = df[df[columns[-1]] == variant]
        # vol (particles)
        fig.add_trace(go.Scatter(x=filtered_df[columns[0]], y=filtered_df[columns[2]],
                    name="(J-method) " + variant),
                    secondary_y=False)
        # vol (grid)
        fig.add_trace(go.Scatter(x=filtered_df[columns[0]], y=filtered_df[columns[3]],
                    name="(Density-method) " + variant),
                    secondary_y=False)
    
    # manipulator position (Only add it for one variant)
    filtered_df = df[df[columns[-1]] == variant_names[0]]
    fig.add_trace(go.Scatter(x=filtered_df[columns[0]], y=filtered_df[columns[1]],
                  fill='tozeroy', name="Manipulator Position"),
                  secondary_y=True)
    fig.update_traces(marker=dict(size=3))
    # fig.update_scenes(aspectmode='data')
    fig.update_layout(title="Volume Preservation Experiment")
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Manipulator Position", secondary_y=True)

    fig.add_annotation(text=annotation, 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0,
                        bordercolor='black',
                        borderwidth=1)
    fig.show()

    if save_output:
        fig.write_html(f"{base_filename}_result.html")


annotation = f'Number of particles: {last_cfg_sim.n_particles} <br>' + \
            f'Ground friction: {last_cfg_sim.ground_friction} <br>' + \
            f'Manipulator friction: {0.9} <br>' + \
            f'Gravity: {last_cfg_sim.gravity} <br>' + \
            f'Action magnitude: {action_magnitude} <br>' +\
            f'Action lowest height: {action_lowest_height} <br>' + \
            f'Yield stress: {last_cfg_sim.yield_stress} <br>' + \
            f'dt: {last_cfg_sim.dt_override} <br>' + \
            f'Episode length: {episode_length} <br>'

now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
base_filename = f"volume_preservation_{date_time}"

if save_output:
    df.to_pickle(f"{base_filename}_dataframe.pkl")
    text_file = open(f"{base_filename}_metadata.txt", "w")
    text_file.write(annotation)
    text_file.close()

create_interactive_plot(df, annotation, save_output, base_filename)

