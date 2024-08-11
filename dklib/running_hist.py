import numpy as np



class hist:
    def __init__(self,bins):
        self.bins = bins 
        self.counts = np.zeros(np.shape(bins)[0]-1,dtype=np.int64) #do we want an integer type here? 
        self.timeavg = avg() #whatever our timelike parameter is 
        self.avg = avg()
    
    @classmethod
    def from_list(cls,histlist):
        newhist = hist(histlist[0].bins)
        weightedMean = 0.
        weightedVar  = 0.
        weightedT    = 0.
        weightedTVar = 0.
        N = np.int64(0)
        for h in histlist:
            if(np.sum(h.counts) == 0):
                continue
            N+=h.avg.N
            weightedMean+=h.avg.mean()*h.avg.N
            weightedVar+=h.avg.var()*h.avg.N
            weightedT+=h.timeavg.mean()*h.timeavg.N
            weightedTVar+=h.timeavg.var()*h.timeavg.N
            newhist.counts+=h.counts 
        newhist.avg.K = weightedMean/N 
        newhist.avg.Ex2_shifted = (weightedVar/N)*(N-1)
        newhist.avg.N = N
        newhist.timeavg.K = weightedT / N 
        newhist.timeavg.Ex2_shifted = (weightedTVar/N)*(N-1)
        newhist.timeavg.N = N 
        return newhist


    def addDat(self,dat,timeparam,destructive=False):
        tempcounts,_ = np.histogram(dat,self.bins)
        self.counts+=tempcounts
        self.timeavg.addDat(np.array([timeparam]))
        self.avg.addDat(dat,destructive)

    def clear(self):
        self.counts[:] = 0 
        self.timeavg.clear() 
        self.avg.clear()



class avg:
    def __init__(self):
        self.K = 0 #offset during running average. 
        self.N = np.int64(0)
        self.Ex_shifted = 0. #expectation of x, shifted
        self.Ex2_shifted = 0. #expectation of x^2, shifted

    @classmethod
    def from_list(cls,avg_list):
        newavg = avg()
        N = np.int64(0)
        weighted_var  = 0 
        weighted_mean = 0
        for av in avg_list:
            N+=av.N 
            weighted_var+=av.var()*av.N
            weighted_mean+=av.mean()*av.N
        newavg.N = N
        newavg.Ex2_shifted = (weighted_var / N) *(N-1)
        newavg.Ex_shifted = 0.
        newavg.K = weighted_mean / N

    #adds data, shifting in place if destructive = True, using a shift parameter set to the first data's mean. 
    #destructive seems to have no effect if we pass dat as a view, e.g. dat[dat>0.5] for example. 
    def addDat(self,dat,destructive=False):
        numentries = np.int64(np.size(dat)) - np.sum(np.isnan(dat))
        if(self.N == 0 and numentries>0):
            self.K = np.nanmean(dat) 
        self.N += numentries
        if(destructive):
            shifteddat = np.subtract(dat,self.K,out=dat)#shifteddat points to dat, skipping a copy, but this process damages the data in dat. 
        else:
            shifteddat = np.subtract(dat,self.K) #produces a copy. 
        self.Ex_shifted+=np.nansum(shifteddat)
        np.multiply(shifteddat,shifteddat,out=shifteddat)
        self.Ex2_shifted+=np.nansum(shifteddat)
    
    def mean(self):
        return self.K + self.Ex_shifted/self.N
    
    def var(self):
        #using N-1 for bessel's correction produces an unbiased estimator of var from finite samples...
        return (self.Ex2_shifted - (self.Ex_shifted*self.Ex_shifted) / self.N) / (self.N-1)

    def clear(self):
        self.K = 0
        self.N = np.int64(0)
        self.Ex2_shifted = 0.
        self.Ex_shifted = 0.

def mpi_collate_histogram(hist):
    from mpi4py import MPI
    from dklib.mpi_utils import is_root

    mean = np.array(hist.avg.mean()) * hist.avg.N
    var = np.array(hist.avg.var()) * hist.avg.N
    N = np.array(hist.avg.N,dtype = np.int64)
    Ntot = np.array(0,dtype=np.int64)
    weightedMean = np.array(0.0)
    weightedVar = np.array(0.0)
    dest = None 
    if(is_root()):
        dest = np.zeros(shape=hist.counts.shape, dtype = hist.counts.dtype)
    MPI.COMM_WORLD.Reduce(hist.counts,dest,op=MPI.SUM,root = 0)
    MPI.COMM_WORLD.Reduce(mean,weightedMean,op=MPI.SUM)
    MPI.COMM_WORLD.Reduce(var,weightedVar,op=MPI.SUM)
    MPI.COMM_WORLD.Reduce(N,Ntot,op=MPI.SUM)

    if(is_root()):
        N = Ntot 
        hist.counts = dest 
        hist.avg.Ex_shifted = 0.0
        hist.avg.K = weightedMean/N
        hist.avg.Ex2_shifted = (weightedVar/N)*(N-1)
        hist.avg.N = N
