import pandas as pd
import numpy as np
import plotly.express as px

base_filename = "material_properties_E_nu_2023-02-09_12-04-08"

def create_interactive_plot(df: pd.DataFrame, annotation: str, save_output: bool=False, base_filename: str=""):
    df = df.apply(pd.to_numeric, errors='ignore')
    columns = list(df.columns)
    # eps = 0.01
    # df = df[(df['x'] < step+eps) & (df['x'] > step-eps)]
    fig = px.scatter_3d(data_frame=df,
                        x=columns[2], y=columns[0], z=columns[1],
                        color=columns[3],
                        title="Material Properties Experiment: Final Deformed Configurations",
                        )
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
    # steps = []
    # eps = 0.01
    # for i in range(0, 100):
    #     target = i/100
    #     step = dict(
    #         method="update",
    #         args=["data_frame", df[(df['x'] < target+eps) & (df['x'] > target-eps)]],
    #         value=target,
    #     )
    #     steps.append(step)

    # sliders = [dict(
    #     active=0.5,
    #     currentvalue={"prefix": "Slice: "},
    #     pad={"t": 50},
    #     steps=steps
    # )]

    # fig.update_layout(
    #     sliders=sliders
    # )

    fig.show()

    if save_output:
        fig.write_html(f"base_filename_results.html")

# Retrieve saved data
df_filename = f"{base_filename}_dataframe.pkl"
metadata_filename = f"{base_filename}_metadata.txt"
df = pd.read_pickle(df_filename)
with open(metadata_filename) as f:
    annotation = f.read()

create_interactive_plot(df, annotation)