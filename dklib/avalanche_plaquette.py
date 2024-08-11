from dolfin import assemble,dx,ds
import numpy as np
#Contains a data structure for holding statistics about a single avalanche.

class Avalanche:
    def __init__(self,measurer, numsites=1, numdyn=1):
        self.numSites=numsites
        self.numset = 0
        self.measurer = measurer
        self.values = np.zeros([2,measurer.N])
        self.values_dyn = np.zeros([numdyn]) #measures taken during the avalanche.


    def setMeasureables(self,Mat):
        vals = self.measurer.measure(Mat)
        if(Mat.cmap.isroot):
            self.values[self.numset,:] = vals
        self.numset +=1 

#measures properties of the EPM material, before and after the avalanche
class epm_plaq_measurer:
    def sig_xy_avg(Mat):
        if(Mat.cmap.isroot):
            return np.mean(Mat.plaq_sig_xy) 

    def x_avg(Mat):
        if(Mat.cmap.isroot):
            return np.mean(Mat.plaq_x)

    def eps_xy_avg(Mat):
        if(Mat.cmap.isroot):
            return np.mean(Mat.plaq_eps_xy) 
            
    #currently inefficient -- can be made faster if we introduce caching? 
    def eigenstrain_xy_avg(Mat):
        array = np.zeros(Mat.cmap.num_plaq)
        Mat.Wmap.plaquette_average_mpi(Mat.eigenstrain.vector().get_local(),array,1)
        if(Mat.cmap.isroot):
            return np.mean(array) 

    def x_min(Mat):
        if(Mat.cmap.isroot):
            return np.min(Mat.plaq_x)
    
    def loadingParam(Mat):
        if(Mat.cmap.isroot):
            return Mat.loadingParam

    MeasuresPairs = [
         ('sig_xy',sig_xy_avg),
         ('x',x_avg),
         ('eps_xy',eps_xy_avg),
         ('eigenstrain_xy',eigenstrain_xy_avg),
         ('x_min',x_min),
         ('loadingParam',loadingParam)]
    Measures = dict(MeasuresPairs)
    MeasureNames = [ p[0] for p in MeasuresPairs] 
    MeasureFuncs = [ p[1] for p in MeasuresPairs]
    
    #class level:
    def __init__(self,measureNames):
        self.measures = [ epm_plaq_measurer.Measures[name] for name in measureNames]
        self.N = np.size(self.measures)
    
    def measure(self,Mat):
        #measures an epm_plaquette class using the given measures
        vals = np.zeros(self.N) 
        for i in range(0,self.N):
            vals[i] = self.measures[i](Mat)
        return vals 