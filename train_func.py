import sys
from multiprocessing import Process

print(sys.path)
from train_3d_tiles_dash import train

''' remember to use conda env dash3d '''
def fx(params):
    
    for k, v in params:
        print(k, v)

    # try:
    #     p = mp.Process(target=train, args=(npath, spath, None))
    #     p.start()
    #     #p.join() 
    #     return "running " + npath
    # except:
    #     return value + " path does not exist"