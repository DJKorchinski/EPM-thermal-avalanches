import numpy as np 

class histNd:
    def __init__(self,binlist):
        self.Nd = np.shape(binlist)[0]
        self.Nbins = np.array([np.size(b)-1 for b in binlist])
        self.binedges = binlist
        self.counts = np.zeros(self.Nbins,dtype=np.int64)
        self.misses = 0#count the number of times that addDat missed?

    
    def addData(self,datalist,weights=None):
        counts,edges = np.histogramdd(np.array(datalist).transpose(),self.binedges,weights=weights)
        self.counts+=np.int64(counts)
        self.misses+=(np.size(datalist[0]) - np.sum(counts))

    @classmethod 
    def fromHist2d(cls,src):
        h = histNd([src.bins1,src.bins2])
        h.counts[:,:] = src.counts[:,:] 
        return h 

#project the pdf onto dims listed by dims. dims = None => all dimensions.
def normHist(hist,dims=None):
    if(dims is None):
        dims = np.arange(hist.Nd)
    
    sumdims = np.delete(np.arange(hist.Nd,dtype=np.int32), dims)
    pdf = np.zeros(shape=hist.Nbins[dims],dtype=np.float64)
    np.sum(hist.counts, axis=tuple(sumdims),out = pdf,dtype=np.int64)
    pdf /= np.sum(hist.counts)

    pdfdims = np.arange(np.size(dims))
    for i,d in enumerate(dims):
        binwids = hist.binedges[d][1:] - hist.binedges[d][:-1]
        otherdims = np.delete(pdfdims,i)
        # print(i,d,pdfdims,otherdims)
        pdf /= np.expand_dims(binwids,axis=tuple(otherdims))
    return pdf 





import copy 
def sliceHist(src,axis,slicemin,slicemax = None):
    binedges = src.binedges[axis]
    if(slicemax is None):
        slicemax = slicemin
    minind = np.clip(np.searchsorted(binedges,slicemin)-1,0,binedges.size-1)
    maxind = np.clip(np.searchsorted(binedges,slicemax)-1,0,binedges.size-1)
    return sliceHistInd(src,axis,minind,maxind)

    
def sliceHistInd(src,axis,minind,maxind):
    hist = copy.deepcopy(src)    
    tup = [np.s_[:]]*hist.Nd
    tup[axis] = np.s_[:minind]
    hist.counts[tuple(tup)] = 0
    tup[axis] = np.s_[maxind+1:]
    hist.counts[tuple(tup)] = 0
    return hist



def mpi_collate_histogram(hist):
    from mpi4py import MPI
    from dklib.mpi_utils import is_root
    dest=None
    missdest = None 
    if(is_root()):
        dest = np.zeros(shape = hist.counts.shape,dtype = hist.counts.dtype)
        missdest = np.zeros(1,np.int64)
    MPI.COMM_WORLD.Reduce(hist.counts,dest,op=MPI.SUM,root=0)
    MPI.COMM_WORLD.Reduce(np.array([hist.misses],dtype=np.int64),missdest,op=MPI.SUM,root=0)
    if(is_root()):
        hist.counts = dest
        hist.misses = missdest[0]