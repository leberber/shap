
import dash_mantine_components as dmc
from dash import html, callback,  Input, Output, State, ALL, no_update, ctx, dcc
import pandas as pd
from dash_iconify import DashIconify
from components import colors


def avsb_definitions(id :str) -> html:
    return html.Div(
                    className = 'avsb-definition-selects-div',
                    children=[
                        html.Div(
                            style={'padding':'17px', 'position': 'relative'},
                            children = [
                                dmc.Select(
                                    label="Select Feature",
                                    id = f'{id}-feature-select',       
                                    ),

                                dmc.SegmentedControl(
                                        fullWidth=True,
                                        data=['==', '!=', '>','>=', '<=', '<'],
                                        id = f'{id}-operator-select',
                                        value = '==',
                                        my=10,
                                        style={'visibility':'hidden'}
                                    ),

                                dmc.Select(
                                    label="Select Category",
                                    id =f'{id}-category-select',
                                    mb = 10,
                                    style={'display':'none'}
                                ),

                                dmc.NumberInput(
                                    id = f'{id}-number-input',   
                                    hideControls = True,
                                    style={'width':'100%', 'display':'none'},
                                    mb = 10,
                                    label=html.Div(   
                                            style={ 'display': 'flex'}, 
                                            children =[
                                                dmc.Text("Input Range",pr=5),
                                                html.Div(
                                                    style={'display':'flex', 'position': 'absolute', 'right':17},
                                                    children = [
                                                        dmc.Text("Min -> ", color="rgba(12, 192, 223, 1)",  mr='3px'),
                                                        dmc.Text( None, color="gray", mr='6px', id = f'{id}-min'),
                                                        dmc.Text( ' |', color="red", mr='6px'),
                                                        dmc.Text("Max -> ", color="rgba(12, 192, 223, 1)", mr='3px'),
                                                        dmc.Text(None, color="gray", id = f'{id}-max')
                                                    ]),        
                                    ]),
                                ), 
                                dmc.Group(
                                    position='right',
                                    mb=10,
                                    children = [
                                        dmc.Button(
                                            "Add Filter",
                                            variant="light",
                                            leftIcon=DashIconify(icon="fa:plus", color="rgba(12, 192, 223, 1)"),
                                            id = f'{id}-add-filter-btn',
                                            style={'display':'none'}
                                        ),
                                    ]
                                ),
                                html.Div(id = f'{id}-added-filters'),
                                dcc.Store(id=f'{id}-filters-data', data= []),
                            ]
                        )
                    ]
            )

avsb = dmc.Paper(
    px = 12,
    pt = 30,
    h = '100%',
    children = [
        dmc.Grid(
                justify="center",
                align="stretch",
                h = '100%',
            children=[

                dmc.Col(
                    avsb_definitions('a'),
                    h = '100%',
                        bg = colors['gray_bg'],
                    span=3, 
                ),

                dmc.Col(
                    pos = 'relative',
                    className='ab-definitions',
                    h = '100%',
                    span=6,
                        children =[    
                            dmc.LoadingOverlay(
                                        html.Div(id='avsb-waterfall'),        
                                loaderProps={"variant": "dots", "color": "orange", "size": "xl"},
                            ),  
                            dmc.Button(
                                    "Run",   variant="default", id="run-ab", pos = 'absolute', bottom= "-40px", right = 12,
                                    rightIcon=DashIconify(icon="ion:caret-forward-circle-outline", color=colors['green'], width=25),
                            ),  
                    ]
                ),

                dmc.Col(
                    avsb_definitions('b'),
                    span=3,
                    bg = colors['gray_bg'],
                ),   
            ],
            gutter="xl",
        ),   
    ]
)

tabs = dmc.Paper(
    display= 'none',
    id = 'tabs-paper',
    mx='10%',
    radius=20,
    shadow = 'xs',
    mt=10,
    p=20,
    mb=1000,
    h= '64vh',
    children=[
        dmc.Tabs(
            h= '90%',
            children = [
                dmc.TabsList(
                    position = 'center',
                    children =[
                        dmc.Tab("Importance", icon=DashIconify(icon="fa6-solid:bars-staggered",  width=20), value="importance"),
                        dmc.Tab("A/B", icon=DashIconify(icon="fluent-emoji-high-contrast:ab-button-blood-type", width=20), value="avsb"),
                        dmc.Tab("Impact", icon=DashIconify(icon="iconamoon:category-fill",  width=20),value="impact"),
                    ]
                ),
                dmc.TabsPanel(
                    value="importance",
                    p = 'auto',
                    m = 'auto',
                    w= '80%',
                    id= 'shap-importance'
                ),
                dmc.TabsPanel(value="impact"),
                dmc.TabsPanel(avsb,value="avsb",  h= '93%',),
            ],
            value="avsb",
            variant = 'pills',
        )
    ]
)

def avsb_layout_control(_id ):
    @callback(
        Output(f'{_id}-operator-select', 'style'),
        Output(f'{_id}-category-select', 'data'),
        Output(f'{_id}-category-select', 'value'),
        Output(f'{_id}-category-select', 'style'),
        Output(f'{_id}-number-input', 'style'),
        Output(f'{_id}-number-input', 'value'),
        Output(f'{_id}-min', 'children'),
        Output(f'{_id}-max', 'children'),
        Output(f'{_id}-add-filter-btn', 'style'),

        State('model-features', 'value'),

        Input(f'{_id}-feature-select', 'value'), 
        prevent_initial_call=True,         
    )
    def _avsb_layout_control(features, feature):
        df= pd.read_parquet('base.gzip')
        features = df.dtypes.apply(lambda x: x.name).to_dict()
        if not feature:
            return no_update
        
        if features[feature] =='object':
            data = [str(c) for c in  df[feature].unique()]

            return {'visibility':'visible'}, data, None, {'display':'block'},  {'display':'none'}, None, None, None, {'display':'block'}

        _min = df[feature].min()
        _max = df[feature].max()
        return {'visibility':'visible'}, [], None, {'display':'none'}, {'display':'block'}, None, str(round(_min, 2)), str(round(_max, 2)), {'display':'block'}


def avsb_add_filters(_id):
    @callback(
                Output(f'{_id}-filters-data', 'data', allow_duplicate=True),
                Output(f'{_id}-added-filters', 'children', allow_duplicate=True),

                State(f'{_id}-feature-select', 'value'),
                State(f'{_id}-operator-select', 'value'),
                State(f'{_id}-category-select', 'value'),
                State(f'{_id}-number-input', 'value'),
                State(f'{_id}-filters-data', 'data'),

                Input(f'{_id}-add-filter-btn', 'n_clicks'),  
                prevent_initial_call = True       
    )
    def _avsb_add_filters(feature, aporator, category, num_input, filters, add_filter):
        # print('feature', feature, 'aporator', aporator, 'category',category,  'num_input', num_input,   add_filter, 'add_filter')
        inputs = [feature, aporator, category, num_input]

        if category:
            my_filter = f"{feature} {aporator } '{category}'"
        else:
             my_filter = f"{feature} {aporator } {num_input}"

        filters.append(my_filter)

        filter_prisms = []
        for idx, fil in enumerate(filters):
            _group = html.Div(
                        style={'display':'flex', 'backgroundColor' :'rgba(248, 249, 250, 0.65)'},
                        children=[
                            dmc.Prism( fil, language="sql", noCopy = True,className='filter-prism'),
                            dmc.ActionIcon(
                                DashIconify(icon="emojione-v1:heavy-minus-sign", width=20),
                                variant="subtle",
                                id={"type": f"{_id}-remove-filter-btn", "index": idx},
                                style={'display':'flex', 'position': 'absolute', 'right':17, 'verticalAlign': 'center'},
                            ),
                        ]
                    )
            filter_prisms.append(_group)

        return filters, filter_prisms


def avsb_remove_filters(_id):
    @callback(
        Output(f'{_id}-added-filters', 'children'),
        Output(f'{_id}-filters-data', 'data'),
        State(f'{_id}-filters-data', 'data'),
        Input({"type": f"{_id}-remove-filter-btn", "index": ALL}, "n_clicks"),
        prevent_initial_call = True       
    )
    def _avsb_remove_filters(filters, filter_btns):
        index = eval(ctx.triggered[0]['prop_id'].replace('.n_clicks', ''))['index']
        value = ctx.triggered[0]['value']

        if value:
            filters.pop(0)
            filter_prisms = []
            for idx, fil in enumerate(filters):
                _group = html.Div(
                        style={'display':'flex', 'backgroundColor' :'rgba(248, 249, 250, 0.65)'},
                        children=[
                            dmc.Prism( fil, language="sql", noCopy = True,className='filter-prism'),
                            dmc.ActionIcon(
                                DashIconify(icon="emojione-v1:heavy-minus-sign", width=20),
                                variant="subtle",
                                id={"type": f"{_id}-remove-filter-btn", "index": idx},
                                style={'display':'flex', 'position': 'absolute', 'right':17, 'verticalAlign': 'center'},
                            ),
                        ]
                    )
        
                filter_prisms.append(_group)
            return filter_prisms, filters
        return no_update, filters
