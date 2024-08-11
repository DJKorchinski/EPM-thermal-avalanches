import dklib.mpi_utils
import numpy as np


class avalanche:
    NUM_VALUES = 6
    T_INDEX = 0
    SIG_INDEX = 1
    ST_COUNT_INDEX = 2
    XMIN_INDEX = 3
    SIG_R_INDEX = 4
    CUMULATIVE_PLASTIC_STRAIN_INDEX = 5
    def __init__(self,mf=None):
        self.values = np.zeros(shape = (2,avalanche.NUM_VALUES))
        if(not (mf is None)):
            self.update(mf)
        
    def update(self,mf,index = 0,st_count = -1):
        self.values[index,avalanche.T_INDEX] = mf.t
        self.values[index,avalanche.SIG_INDEX] = np.mean(mf.sigma)
        self.values[index,avalanche.CUMULATIVE_PLASTIC_STRAIN_INDEX] = np.mean(mf.cumulative_eigenstrain)
        solid_inds = mf.solid_site_inds()
        if(solid_inds.size > 0):
            self.values[index,avalanche.XMIN_INDEX] = np.min(mf.x[solid_inds])
        else: 
            self.values[index,avalanche.XMIN_INDEX] = -1.0
            dklib.mpi_utils.printmpi('had a case of all sites fluidized.')
            dklib.mpi_utils.printmpi('tlast: ' ,mf.t - mf.tlast)
            dklib.mpi_utils.printmpi('tlast: ' ,mf.t - mf.tlast > mf.fluidization_time())
        self.values[index,avalanche.SIG_R_INDEX] = np.mean(mf.sigma_res)
        if(st_count >=0):
            self.values[index,avalanche.ST_COUNT_INDEX] = st_count 
        #can't update ST_COUNT_INDEX
    
    def list2np(list):
        N = np.size(list)
        arr = np.zeros(shape = (N,2,avalanche.NUM_VALUES))
        for i,av in enumerate(list):
            arr[i] = av.values
        return arr 



class STRecord:
    NUM_VALUES = 5
    T_INDEX = 0
    SIG_INDEX = 1
    SIG_RES_INDEX = 2
    AV_INDEX = 3 
    PLASTIC_STRAIN_INDEX = 4
    def __init__(self,mf=None):
        self.values = np.zeros(shape = (STRecord.NUM_VALUES))
        if(not (mf is None)):
            self.update(mf)
    
    def update(self,mf,avno=0):
        self.values[STRecord.T_INDEX] = mf.t
        self.values[STRecord.SIG_INDEX] = np.mean(mf.sigma)
        self.values[STRecord.SIG_RES_INDEX] = np.mean(mf.sigma_res)
        self.values[STRecord.AV_INDEX] = avno
        self.values[STRecord.PLASTIC_STRAIN_INDEX] = np.mean(mf.cumulative_eigenstrain)
    
    def list2np(list):
        N = np.size(list)
        arr = np.zeros(shape = (N,STRecord.NUM_VALUES))
        for i,av in enumerate(list):
            arr[i] = av.values
        return arr

