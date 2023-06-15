from dash import Dash, html, callback,  Input, Output, State, no_update, ctx, dcc, ALL, get_asset_url
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import matplotlib.pyplot as pl
import shap
import pickle
import numpy as np
import pandas as pd

from math import exp
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import base64
import pandas as pd
import io

def icon(icon, color='white', height=20):
    return DashIconify(icon=icon, height=height, color= color)

def encoder(df, encodings):
    encodings_ = {}
    for key, value in encodings.items():
        encodings_.setdefault(value, []).append(key)
    df[encodings_['Ordinal']] = OrdinalEncoder().fit_transform(df[encodings_['Ordinal']])
    for feat in encodings_['Nominal']:
        df[feat] = LabelEncoder().fit_transform(df[feat])
    return df

def shap_(df, features, target):
    X = df[features]
    Y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy on test data: %.2f%%" % (accuracy * 100.0))
    auc= roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("AUC on test data", round(auc,2))
    
    explainer = shap.TreeExplainer(model)
    baseValue = explainer.expected_value[0]
    _shap = explainer.shap_values(X)
    shap.summary_plot(_shap, X, max_display=10,show=False)
    path = pl.savefig("assets/shap_summary.svg",dpi=700)
    
    _shap = pd.DataFrame(_shap, columns = X.columns, index=df.index)
    _shap['baseValue'] = baseValue
    _shap['sumShap'] = _shap[_shap.columns].sum(axis=1)
    _shap['Y'] = Y
    _shap['P']=_shap.apply(lambda x: 1/(1+ exp(-x['sumShap'])), axis=1)
    _shap['R'] = _shap['Y'] - _shap['P']
    return _shap

def stepper(colors):
    return dmc.Paper(
        id= 'stepper-paper',
        bg =colors['white'],
        m='20% 30% 0% 30%',
        radius=15,
        shadow = 'xs',
        p = 20,
        children =[
            dmc.Stepper(
                id="stepper",
                active=0,
                breakpoint="lg",
                children=[
                    dmc.StepperStep(
                        label="First Step",
                        description="Upload Data",
                        icon=icon(icon="basil:upload-solid"),
                        completedIcon=icon(icon="basil:upload-solid"),
                        children=dmc.Center(
                                    pt = 20,
                                    children =[
                                        dcc.Upload(
                                            id='upload-data',
                                             multiple=True,
                                            children=dmc.Paper(
                                                bg =colors['gray_bg'],
                                                p = 40,
                                                radius=15,
                                                children = [
                                                    'Drag and Drop or Select Files',
                                            ]),
                                        )
                                ]),
                    ),

                    dmc.StepperStep(
                        label="Second step",
                        description="Categorical Encoding",
                        icon=icon(icon="charm:binary", color=colors['tomato']),
                        progressIcon=icon(icon="charm:binary"),
                        completedIcon=icon(icon="charm:binary"),
                        children=dmc.Center(id = 'category-encoding',  pt = 20)
                    ),

                    dmc.StepperStep(
                        label="Third Step",
                        description="Features Selection",
                        icon=icon(icon="ph:selection-all-bold", color=colors['tomato']),
                        progressIcon=icon(icon="ph:selection-all-bold"),
                        completedIcon=icon(icon="ph:selection-all-bold"),
                        children=dmc.Center(id = 'feature-selection',  pt = 20,)
                    ),

                    dmc.StepperCompleted(
                        children=dmc.Text(
                            "Completed, click back button or run to start analysis",
                            align="center",
                        )
                    ),
                ],
            ),
            dmc.Group(
                position="center",
                mt="xl",
                children=[
                    dmc.Button("Back",  variant="default", id="back", n_clicks = 0, leftIcon=DashIconify(icon="ion:caret-back-outline", color=colors['tomato'],  width=20)),
                    dmc.Button("Next",  variant="default", id="next", n_clicks = 0, rightIcon=DashIconify(icon="ion:caret-forward-outline", color=colors['tomato'], width=20)),
                    dmc.Button("Run",   variant="default", id="run", rightIcon=DashIconify(icon="ion:caret-forward-circle-outline", color=colors['green'], width=25), display='none'),
                ],
            ),
        ]
    )




def stepper_navigation_callback():
    @callback(
        Output("stepper", "active",  allow_duplicate=True),
        Output("next", "display"),
        Output("run", "display"),
        Output("back", "n_clicks"),
        Output("next", "n_clicks"),
        
        Input("back", "n_clicks"),
        Input("next", "n_clicks"),
        State("stepper", "active"),
        prevent_initial_call=True,
    )
    def _stepper_navigation_callback(back, next_, current):

        min_step = 0
        max_step = 3
        active = 1
        
        step = current if current is not None else active

        if ctx.triggered_id == "back":
            step = step - 1 if step > min_step else step
        else:
            step = step + 1 if step < max_step else step

        if next_ - back <= min_step:
            return step, 'block', 'none', 0, 0
        
        if next_ - back == max_step-1:
            return step, 'none', 'block', no_update, no_update

        return     step, 'block', 'none', back, next_


  
def stepper_content_callback():
    @callback(
        Output('category-encoding', 'children'),  
        Output('feature-selection', 'children'),  
        Output("stepper", "active"),
        
        Input('upload-data', 'contents'),
        prevent_initial_call=True,
    )
    def _stepper_content_callback(content):

            if content :
                def parse_contents(contents):
                    _, content_string = contents[0].split(',')
                    decoded = base64.b64decode(content_string)
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                    return df

                df =  parse_contents(content)
                
                categorical = df.select_dtypes(include=['object']).columns.tolist()
                categorical_features = []
                for feat in categorical:
                    feature_group = dmc.Group(
                        p = 5,
                        children = [
                            dmc.Text(feat, style={"width": 90}), 
                            dmc.SegmentedControl(id={"type": "feature-encoding", "index": feat}, data=[ 'Nominal', 'Ordinal'], value = 'Nominal') 
                        ]
                    )
                    categorical_features.append(feature_group)

                def chips (_id, my_chips,  multiple= True) :
                    return html.Div(
                        children = [
                            dmc.ChipGroup(
                                id=_id,
                                multiple=multiple,
                                children = [ 
                                    dmc.Chip( x, value=x, variant="filled", color = 'green') for x in my_chips
                                ]
                            )
                        ]
                    )
                feature_selection = dmc.Grid(
                    children=[
                        dmc.Col(
                            span =6,
                            children=[
                                dmc.Text('Select  Featues', align='center', p=10),
                                chips ('model-features', sorted(df),  multiple= True) 
                            ]
                        ),

                        dmc.Col(
                            span =6,
                            children=[
                            dmc.Text('Select  Target', align='center', p = 10),
                            chips ('target', sorted(df),  multiple= False)
                            ]
                        )
                    ]
                )

                df.index.name='_key'
                df.to_parquet('base.gzip',compression='gzip')

                return html.Div(categorical_features), feature_selection , 1
            return no_update

def get_stepper_content_callback():
    @callback(
        Output('stepper-paper', 'style'), 
        Output('tabs-paper', 'display'), 
        Output("a-feature-select", "data"),
        Output("b-feature-select", "data"),
        Output("shap-importance", "children"),
        
        State({"type": f"feature-encoding", "index": ALL}, "value"),
        State('model-features', 'value'),
        State('target', 'value'),
        Input('run', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _stepper_content_callback(encodings, features, target,  run): 

        encodings =  {_id['id']['index']: _id['value'] for _id in ctx.states_list[0]}

        df = pd.read_parquet('base.gzip')
        df = encoder(df, encodings)
        _shap = shap_(df, features, target) 
        _shap.to_parquet('shap.gzip',compression='gzip')

        image= dmc.Paper(
            h = '90%',
            children=[
                dmc.Text("Shap Feature Impact", size= 24, align="center", mt='60px'),
                dmc.Center( 
                    children=[
                        dmc.Image(
                            src='assets/shap_summary.svg', 
                        )
                    ]
                )
            ]
        )
     
        return {'marginTop': 100, 'marginBottom': 55 }, 'block', features, features, image
