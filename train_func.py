import time
import sys
import re
from multiprocessing import Process

from train_3d_tiles_dash import train
import sendmail

''' remember to use conda env dash3d '''

def fx(params, is_test=True):
   
    if is_test: 
        func = dummy
    else:
        func = train

    kwargs = {'project_name':params['project_name'],
                'train_folder':params['train_folder'],
                'validation_folder':params['validation_folder'],
                'image_extension':params['image_extension'],
                'force_z':params['force_z'],
                'batch_size':params['batch_size'],
                'channel_zero_index':params['channel'] - 1,
                'epochs':params['epochs']}
    
    kwargs['tile_size'] = (params['tile-size-x'],
                            params['tile-size-y'],
                            params['tile-size-z'])
    
    
    kwargs['overlap_fraction'] = (params['x-overlap'],
                                    params['y-overlap'],
                                    params['z-overlap'])

    print(kwargs)
    try:
        p = Process(target=func, kwargs=kwargs)
        p.start()
        #p.join() 
        return p #"running " + npath
    except:
        return " path does not exist"

def notify(params):
    to = params['email']
    match = re.match(
        '^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$',
        to)

    if match is None:
        print("No email address")
        return
    
    t = time.asctime()
    text = f"""The Unet 3D spot training completed at {t}""" 
    sp = '\n'.join([f'{k}:{v}'
                    for k, v in params.items()])

    text = text + '\n\n' + sp
    msg = sendmail.message("Training complete", text) 

    try:
        e = Process(target=sendmail.send, args=(msg, to)) 
        e.start()
    except:
        print("problem with email")

    return

def dummy(**kwargs):
    for i in range(20):
        print(i, time.asctime(), kwargs['project_name'])
        time.sleep(1)