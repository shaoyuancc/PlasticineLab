import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

base_filename = "material_properties_E_nu_2023-02-15_12-40-27"
should_save = False

def create_tomography_plot(df: pd.DataFrame, annotation: str, save_output: bool=False, base_filename: str=""):
    df = df.apply(pd.to_numeric, errors='ignore')
    columns = list(df.columns)
    variants = list(df.variant.unique())
    fig = go.Figure()

    # Add traces, one for each slider step
    step_size = 1/64 #0.01
    for step in np.arange(0,1,step_size):
        eps = step_size/2
        filtered_df = df[(df['x'] < step+eps) & (df['x'] > step-eps)]
        # Add a trace for each variant
        for v in variants:
            v_filtered_df = filtered_df[filtered_df.variant == v]
            fig.add_trace(go.Scatter3d(
                visible=False,
                x=v_filtered_df[columns[2]], y=v_filtered_df[columns[0]], z=v_filtered_df[columns[1]],
                name=v,
                mode='markers'
            ))

    mid = len(fig.data)//2
    for d in fig.data[mid:mid+len(variants)]:
        d.visible = True
    fig.update_traces(marker=dict(size=3))
    fig.update_scenes(aspectmode='data')
    fig.add_annotation(text=annotation, 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=0,
                        y=0,
                        bordercolor='black',
                        borderwidth=1)
    
    # add the slider to allow slicing
    steps = []
    for i in range(0,len(fig.data),len(variants)):
        step = dict(
            method="update",
            args=[  {"visible":[False] * len(fig.data)}, # update for traces
                    # Material Properties Experiment: y = {i*step_size/len(variants)}
                    {"title":f"Final Deformed Configurations (Tomography) (y = {(i*step_size/len(variants))})"} # update for layout 
                ],
        )
        step["args"][0]["visible"][i:i+len(variants)] = [True]*len(variants)  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=len(steps)//2,
        currentvalue={"prefix": "Slice: "},
        pad={"t": 50},
        steps=steps
    )]

    axis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = step_size
    )

    scene = dict(
        aspectmode  ='data',
        xaxis = axis,
        yaxis = axis,
        zaxis = axis,
   )

    fig.update_layout(
        sliders=sliders,
        scene=scene
    )

    fig.show()

    if save_output:
        fig.write_html(f"{base_filename}_tomography.html")

# Retrieve saved data
df_filename = f"{base_filename}_dataframe.pkl"
metadata_filename = f"{base_filename}_metadata.txt"
df = pd.read_pickle(df_filename)
with open(metadata_filename) as f:
    annotation = f.read()

create_tomography_plot(df, annotation, should_save, base_filename)