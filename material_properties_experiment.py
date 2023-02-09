from plb.envs import  make
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

env_names = [f"BallSquish-v{i}" for i in range(1,6)]
episode_length = 300
v_eps = 1e-3
should_render = False
action_magnitude = 0.05

class Agent():
    def __init__(self, env):
        self.mode = 0
        pass
        
    def get_action(self, state):
        manipulator_state = state[-7:]
        print(manipulator_state) # [xpos, ypos, zpos, 4 rotation states]
        if self.mode == 0: # Descend
            action = [0, -action_magnitude, 0]
            if manipulator_state[1] <= 0.15:
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

for env_name in env_names:
    print(f"Running {env_name}")
    env = make(env_name)
    agent = Agent(env)
    rewards = list()
    
    observation = env.reset()
    for t in range(episode_length):
        print(f"step {t}")
        if should_render:
            env.render()
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
            
    variant_names.append(f"E={env.cfg_sim.E:.0e}, nu={env.cfg_sim.nu:.0e}")
    n = env.cfg.n_observed_particles
    only_particles = observation[:-7].reshape((n,6)) # Remove state of manipulator
    only_positions = only_particles[:,:3] # Only keep positions (throw velocities away)
    only_velocities = only_particles[:,3:] 
    if np.any(np.abs(only_velocities) > v_eps):
        print(f"[ALERT] Final particle velocities not within {v_eps} of 0")
        print(only_velocities)
    final_positions.append(only_positions.flatten())

    last_cfg_env = env.cfg
    last_cfg_sim = env.cfg_sim
    env.close()


columns= ['x','y','z', 'variant']
dfs = []
for i, points_flattened in enumerate(final_positions):
    n = points_flattened.size//3
    xyz = points_flattened.reshape(n, 3)
    obj_label = np.full((n,1),variant_names[i])
    
    xyzobj = np.hstack([xyz, obj_label])
    new_df = pd.DataFrame(data=xyzobj,columns=columns,)
    dfs.append(new_df)
data_frame = pd.concat(dfs, ignore_index=True)
fig = px.scatter_3d(data_frame=data_frame,
                    x=columns[2], y=columns[0], z=columns[1],
                    color=columns[3],
                    title="Material Properties Experiment: Final Deformed Configurations",
                    )

fig.update_traces(marker=dict(size=3))
fig.update_layout(autotypenumbers='convert types')
fig.update_layout(scene_aspectmode='cube')

fig.add_annotation(text=f'Number of particles: {last_cfg_sim.n_particles} <br>' +
                        f'Ground friction: {last_cfg_sim.ground_friction} <br>' + 
                        f'Manipulator friction: {0.9} <br>' + 
                        f'Gravity: {last_cfg_sim.gravity} <br>' + 
                        f'Action magnitude: {action_magnitude} <br>' +
                        f'Yield stress: {last_cfg_sim.yield_stress} <br>' + 
                        f'dt: {last_cfg_sim.dt_override} <br>' + 
                        f'Episode length: {episode_length} <br>', 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0,
                    bordercolor='black',
                    borderwidth=1)
fig.show()
fig.write_html("material_properties_E_nu_results.html")