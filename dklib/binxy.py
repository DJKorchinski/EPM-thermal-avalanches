#performs binning of (x,y) data over xbins
import numpy as np 
def binxy(x,y,xbins):
    xcounts,edges = np.histogram(x,xbins)#
    weights,edges = np.histogram(x,xbins,weights = y)
    mean = weights
    mean[xcounts > 0]  /= xcounts[xcounts > 0]
    return mean     