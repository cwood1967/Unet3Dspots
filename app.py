import os
import time
import socket
import multiprocessing as mp

import dash
from dash import dependencies
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, MATCH, ALL

from app_utils import checktype, checkpath
from train_func import fx, notify
import sendmail

process_dict = {'process':None, 'params':None}
doclink = 'https://webfs/n/core/micro/internal/unet3d/trainingdocs.md'
def inputDiv(label, id, value, *args):
    """Return a dash html div with Label, Input, Label in a row.
    
    Parameters
    ----------
    label : str
            The text beside the input box.
    id :    str
            The id of the html element
    value : The default value of the parameter

    """
    print(label, id)
    div = html.Div([
        html.Label(label, style={'float':'left', 'right-margin':'10px',
                                        'width':'20%'}),
        dcc.Input(id={'type':'input', 'id':id}, value=value,
                type='text', size=30, style={'float':'left', 'width':'30%',
                                            'left_margin':'10px'}),
        html.Label(id={'type':'val', 'id':f'val-{id}'}, children=[],
                   style={'float':'left', 'left-margin':'10px',
                          'width':'30%', 'font-weight':'bold',
                          'color':'#ff00ff'}),
        ], style={'width':'85%','display':'inline-block'})

    return div

"""Dictionary describing parameters {name, id, value, type}."""
inputmap = {
    0:['Project Name', 'input-project_name', 'My Training', str],
    1:['Training Folder', 'input-train_folder', '/n/core/micro/', str],
    2:['Validation Folder', 'input-validation_folder', '/n/core/micro/', str],
    3:['Image Extension', 'input-image_extension', '.tif', str],
    4:['Channel (one based)', 'input-channel', '1', int],
    5:['Tile Size X', 'input-tile-size-x', '128', int],
    6:['Tile Size Y', 'input-tile-size-y', '128', int],
    7:['Tile Size Z', 'input-tile-size-z', '16', int],
    8:['Force Z', 'input-force_z', '32', int],
    9:['Batch Size', 'input-batch_size', '16', int],
    10:['Epochs', 'input-epochs', '300', int],
    11:['X Overlap Fraction', 'input-x-overlap', '0.5', float],
    12:['Y Overlap Fraction', 'input-y-overlap', '0.5', float],
    13:['Z Overlap Fraction', 'input-z-overlap', '0.5', float],
    14:['Email address', 'input-email', '', str],
}

def validate(values):
    """Validate the values in the input widget
    
    Parameters
    ----------
    values : list
             inputs from layout widgets based on input index

    """         
    v = ["" for k in inputmap.keys()] 
    #v[0] = checkpath(values[0]) 
    v[1] = checkpath(values[1]) 
    v[2] = checkpath(values[2]) 
        
    if values[3] not in ['.tif', '.nd2', '.lsm', '.czi']:
        v[3] = "unknown file extension"
   
    v[4] = checktype(values[4], int, "Can't convert to int")
    if not v[4]:
        if int(values[4]) <= 0:
            v[4] = "Channels start at 1"
        
    v[5] = checktype(values[5], int, "Can't convert to int")
    v[5] = checktype(values[5], int, "Can't convert to int")
    v[6] = checktype(values[6], int, "Can't convert to int")
    v[7] = checktype(values[7], int, "Can't convert to int")
    v[8] = checktype(values[8], int, "Can't convert to int")
    v[9] = checktype(values[9], int, "Can't convert to int")
    v[10] = checktype(values[10], int, "Can't convert to int")
    v[11] = checktype(values[11], float, "Can't convert to float")
    v[12] = checktype(values[12], float, "Can't convert to float")
    v[13] = checktype(values[13], float, "Can't convert to float")

    return v

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(id='running', children=[]),
    html.Div(html.H3(f'3D Spot training on {socket.gethostname()}')),
    html.Div([html.H5('Enter parameters for training', id='header'),
             html.A(children="Click here for explaination of parameters",
                    href=doclink, target="_blank")]),
    html.Div(id='input-div',
                children=[inputDiv(*v) for k,v in inputmap.items()]),
    html.Div([
        html.Button("Train", id='button-run', title="Push me"),
        html.Div(id='output', children='Result')
    ])
], id='app')

def get_param_table_div():

    content = [html.Tr([html.Td(k), html.Td(v)])
              for k, v in process_dict['params'].items()]
    table = html.Table(
        [html.Tr([html.Th('Parameter'), html.Th('Value')]),
            html.Tr(content)
        ]
    )
    return table
   

def running_div():
    interval = dcc.Interval(id='interval', disabled=False, interval=2000)
    z = html.Div(get_is_running(),id='status')
    div = html.Div([interval, z,
                    get_param_table_div()],
                   id='app')
    return div

def get_is_running():
    try:
        if process_dict['process'].is_alive():
            return "is running"
        else:
            return "is not running"
    except:
        return "not a process"

@app.callback(
    [Output('app', 'children'),
     Output({'type':'val', 'id':ALL}, 'children')],
    Input('button-run', 'n_clicks'),
    [State({'type':'input', 'id':ALL}, 'value'),
     State('app', 'children')])
def run_training(n_clicks, value, app_children):
    print(n_clicks)
    params = dict()
    v = validate(value)
    if n_clicks is None:
        return app_children, v
    vlen = len([True for i in v if i])
    if vlen > 0:
        return app_children, v
    for i,c in enumerate(value):
        f = inputmap[i][3]
        params[inputmap[i][1][6:]] = f(value[i]) 
    
    process_dict['mail_sent'] = False    
    process_dict['params'] = params
    process_dict['process'] = fx(params, is_test=False)

    return running_div(), v

@app.callback(
    [Output('status', 'children'),
     Output('interval', 'disabled')],
    Input('interval', 'n_intervals')
)
def update_status(n_intervals):
    res = get_is_running()
    t = time.asctime() 

    disabled = False
    if not process_dict['mail_sent']:
        if not process_dict['process'].is_alive():
            process_dict['mail_sent'] = True
            disabled = True
            notify(process_dict['params'])
    
    print(disabled, n_intervals)    
    return f'{t} : {res}', disabled

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=9800)