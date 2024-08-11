import numpy as np 
from numba import jit 

#accepts any scalar, or any list that will 
def listoflists(dimensions):
    lengths = np.array(dimensions,copy=False) if not np.isscalar(dimensions) else np.array([dimensions])
    if(np.size(lengths) > 1):
        return [ listoflists(lengths[1:]) for i in range(0,lengths[0])]
    else:
        return [ [] for i in range(0,lengths[0])]

#A should be a list of one-d lists (or np arrays)
def flattenragged(A,arrdtype=np.int32):
    lens = np.array([np.shape(a)[0] for a in A],dtype=np.int32)
    starts = np.cumsum(lens) - lens
    # flat  = np.zeros(starts[-1]+lens[-1],arrdtype)
    return arrdtype(np.concatenate(A)),lens,starts

class ragged:
    def __init__(self,A,arrdtype=np.int32):
        self.dat,self.lens,self.inds = flattenragged(A,arrdtype)
    
    #probably want to write some nice indexing routines.
    def __getitem__(self,ind):
        #this might return a copy? 
        return self.dat[self.inds[ind]:self.inds[ind]+self.lens[ind]]
    def __iter__(self):
        yield from [ self[i] for i,l in enumerate(self.lens) ]
    





@jit(nopython=True,cache=True)
def sumreduce(L1,L2,L3):
    for i in range(L1.shape[0]):
        L3[L2[i]]+=L1[i]
    return L3

@jit(nopython=True,cache=True)
def sumreduce_strided(L1,L2,L3,offset,stride):
    for j in range(offset,L1.shape[0],stride):
        L3[L2[j]]+=L1[j]
    return L3