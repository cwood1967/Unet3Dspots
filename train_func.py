import time
import sys
from multiprocessing import Process

print(sys.path)
from train_3d_tiles_dash import train


''' remember to use conda env dash3d '''
def fx(params):
    
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
        p = Process(target=train, kwargs=kwargs)
        p.start()
        #p.join() 
        return p #"running " + npath
    except:
        return " path does not exist"
