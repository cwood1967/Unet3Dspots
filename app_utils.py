import os

def checktype(x, f, message="Bad input"):
    """ Check that x can be converted with f.
    
    Parameters
    ----------
    x
        Value to convert  

    f: function
        Function for converting `x`  

    """
    try:
        cx = f(x)
        res = ''
    except:
        res = message 
    return res

def checkpath(p, message="Path does not exist."):
    """Returns a message if path does not exist"""
    if os.path.exists(p):
        res = ''
    else:
        res = message
    return res