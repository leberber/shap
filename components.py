import dash_mantine_components as dmc
from dash import html
import plotly.graph_objects as go
import numpy as np
import pandas as pd


colors = dict(
    tomato ='rgba(255, 99, 71, 1)',
    blue= 'rgba(12, 192, 223, 1)',
    green='rgb(30, 222, 132)',
    gray_bg= 'rgba(243, 244, 245,0.5)',
    white ='white'
)

tick_font_size=16
tick_font_color="rgb(202, 202, 204)"

axis_title_font_size=16
axis_title_font_color="black"

tickfont_family ='Gill Sans (sans-serif)'

legend_font_size=14
legend_font_color="#9e9e9e"

plot_background_color = "white"
paper_background_color = "white"
font_family="verdana, arial, sans-serif"

traces = dict(
            textfont=dict(
        family=tickfont_family,
        size=12,
        color=tick_font_color
    ),
)

gl=dict(

    plot_bgcolor = plot_background_color,
    paper_bgcolor = plot_background_color,
    margin={'t': 50,'r': 20, 'b':20, 'l':20, 'pad':4},
    font=dict(
        family=font_family,
        size=16,
        color=axis_title_font_color
    ),
    

    xaxis=dict(
        tickfont = dict(
            size = tick_font_size,
            color = tick_font_color,
            family = font_family,
            ),
        titlefont = dict(
            size=axis_title_font_size,
            color = axis_title_font_color,
            family = font_family,
        )
    ),

    yaxis = dict(
        tickfont = dict(
            size = tick_font_size,
            color = tick_font_color,
            family = font_family,
        ),
        titlefont = dict(
            size = axis_title_font_size,
            color = axis_title_font_color,
            family = font_family,
        ), 
    ),

 legend= dict(
        font=dict(
        size=legend_font_size,
        color=legend_font_color,
        family = font_family,
        ),
    ),
)

def avsb_df(df, _shap, a_f, b_f, features):

    a_actual, a_residual = float(_shap.loc[df.query(a_f).index][['Y']].mean()), float(_shap.loc[df.query(a_f).index][['R']].mean())
    b_actual, b_residual = float(_shap.loc[df.query(b_f).index][['Y']].mean()), float(_shap.loc[df.query(b_f).index][['R']].mean())
    alloc = b_actual - a_actual - b_residual + a_residual

    _shap = _shap[features]
    A = _shap.loc[df.query(a_f).index].mean()
    B = _shap.loc[df.query(b_f).index].mean()
    
    df = B.subtract(A).to_frame(name='avg_shap_diff').reset_index()
    df['avg_shap_diff_sum'] = df.avg_shap_diff.sum()
    df = df.rename({'index':'features'}, axis=1)
    df['proportion']= (df['avg_shap_diff']/df['avg_shap_diff_sum']) *   100

    df['contribution'] =  df['proportion'] * alloc
    
    df= df[['features', 'contribution']]
    df['measure'] = 'relative'
    df = df.sort_values(by=['contribution'], ascending=False)
    values = df.values.tolist()
    df = df.sort_values(by=['contribution'])
    values.insert(0,   ['A',  a_actual*100,  'absolute'])
    values.append(['A Residuals', - a_residual*100,  'relative'])
    values.append(['B Residuals', b_residual*100, 'relative'])
    values.append(['B',  b_actual*100,  'total'])

    df = pd.DataFrame(values, columns = df.columns)
    df['no_yaxis'] = np.where(df['features'].isin(['A', 'B']), df['contribution']/2, df['contribution'])

    return df

def generate_waterfall(df):
    red = '255, 49, 49'
    green = '126, 217, 87'
    fig  = go.Figure()
    fig.add_trace(go.Waterfall(
        width= [0.8] * len(df),
                x = df['features'], 
                y = df['no_yaxis'],
                measure = df['measure'],
                base = 0,
                cliponaxis= False,
                textfont=dict(
                    family="verdana, arial, sans-serif",
                    size=10,
                    color="rgb(148, 144, 144)"
                ),
                text =  df['contribution'].apply(lambda x: '{0:1.2f}%'.format(x)),
                textposition = 'outside',
                decreasing = {"marker":{"color":f"rgba({red}, 0.7)",  "line":{"color":f"rgba({red}, 1)","width":2}}},
                increasing = {"marker":{"color":f"rgba({green}, 0.7)","line":{"color":f"rgba({green}, 1)", "width":2}}},
                totals     = {"marker":{"color":"rgba(12, 192, 223, 0.7)", "line":{"color":"rgba(12, 192, 223, 1)", "width":2}}},
                connector = {"line":{"color":"rgba(217, 217, 217, 1)", "width":1}},
                ))

    
    end = 0.5 + len(df) -2
    start = 0.5 + len(df) -4
    fig.add_vrect(x0=start, x1=end, fillcolor="rgb(78, 181, 207)", opacity=0.15 , line_color="rgb(78, 181, 207)")

    fig.update_yaxes(ticksuffix = "%")
    fig.update_layout(gl)
    fig.update_traces(traces)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        hovermode=False
        )
    fig.update_layout(
        margin=dict(t=20),
    )

    return fig

def progres_bars (title, labels, values, barwidth = 300):
        bars_labes = []
        bars = []
        bars_text = []
        _max = max(values)
        featues_dict = dict(zip(labels, values))
        featues_dict = dict(sorted(featues_dict.items(), key=lambda item: -item[1]))
        for label, value in featues_dict.items():

            bar_label = str(round(value,2))+'%'
            percent_width =(value/_max)
            print(percent_width)

            bars_labes.append(dmc.Text(label, ta= 'right', weight=200))
            bars.append(dmc.Progress(value = percent_width*100, style={'width':barwidth, 'cursor': 'pointer'}, color ='rgba(255, 99, 71, 1)', className='chart', m =5, size = 20, bg= 'transparent'))
            bars_text.append(dmc.Text(bar_label, weight=200,  style={'marginLeft':-((barwidth+17)-(barwidth * percent_width))}))
           
        return html.Div(
            
            className='barChart',
            children=[
                dmc.Text(title,  weight=300, align="center" , size = 20),
                
                dmc.Group(
                    className='bar-chart-group',
                    children=[
                        
                        html.Div(bars_labes, style={'marginRight':'-15px'}),
                        html.Div(bars),
                        html.Div(bars_text)
                    ]
                ) 
            ]
        )

