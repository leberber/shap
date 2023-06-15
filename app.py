from dash import Dash,  dcc, Input, Output, State, callback
import dash_mantine_components as dmc

import pandas as pd

from stepper import stepper, stepper_navigation_callback, stepper_content_callback, get_stepper_content_callback
from components import generate_waterfall, avsb_df, colors
from avsb import avsb_remove_filters, avsb_layout_control, avsb_add_filters, tabs


app = Dash(__name__, suppress_callback_exceptions=True)


app.layout = dmc.MantineProvider(
    theme={},
    children=[
        stepper(colors),
        tabs
    ]
)

stepper_navigation_callback()
stepper_content_callback()
get_stepper_content_callback()

avsb_layout_control('a')   
avsb_remove_filters('a')
avsb_add_filters('a')

avsb_layout_control('b')   
avsb_remove_filters('b')
avsb_add_filters('b')

@callback(
    Output(f'avsb-waterfall', 'children'),
    State('a-filters-data', 'data'),   
    State('b-filters-data', 'data'),
    State('model-features', 'value'),  
    Input('run-ab','n_clicks'),
    prevent_initial_call = True       
)
def _avsb_run(a_f, b_f, features, run_btn):
    df= pd.read_parquet('base.gzip')
    _shap= pd.read_parquet('shap.gzip')
  
    a_f = ' & '.join(a_f)
    b_f = ' & '.join(b_f)

    waterfall_df = avsb_df(df, _shap, a_f, b_f, features)

    fig = generate_waterfall(waterfall_df)

    return dmc.Paper(
                children=[
                        dmc.Text("Explanation of the Diffrence in Average Predictions of A & B", size= 20, align="center", mt=20 ),
                        dmc.Text("A and B are subsets of data after applied filters",  size= 12,color="gray",align="center",  mt=0, pt=0, mb=0, pb=0),
                        dcc.Graph(figure=fig)
            ])



if __name__ == '__main__':
	app.run_server(
    )
