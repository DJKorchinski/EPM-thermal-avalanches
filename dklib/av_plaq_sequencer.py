#=================
#a small helper library to perform on the fly calculations and averaging of avalanche objects. 
#=================

from abc import ABC, abstractmethod
import dklib.avalanche_plaquette
import numpy as np

class av_sequencer(ABC):
    @abstractmethod
    def addAv(self,prev,curr):
        pass 

class binfunc(av_sequencer):
    #takes in two functions, bins over first function, and averages over the second function in each bin .
    def __init__(self,func1,args1,bins1,func2,args2):
        self.bins = bins1 
        self.counts = np.zeros(len(bins1)-1)
        self.weights = np.zeros(len(bins1)-1)
        self.func1 = func1 
        self.func2 = func2
        self.args1 = args1 
        self.args2 = args2 

    def addAv(self,prev,curr):
        binno = np.digitize(self.func1(prev,curr,*self.args1),self.bins)-1
        # print(self.func1(prev,curr,*self.args1),binno,self.args1)
        if(binno<0):
            return False 
        try:
            fval = self.func2(prev,curr,*self.args2)
            if(not np.isnan(fval)):
                self.counts[binno]+=1
                self.weights[binno]+= fval
            return True
        except IndexError: 
            return False  
    
    def avgs(self):
        return self.weights / self.counts

#This class can probably be a wrapper for binfunc, using the identity function. Just use to quickly plot, e.g. numSites vs. vsstress1
class binxy(binfunc):
    def __init__(self,propind1,propind2,bins1):
        super().__init__(Prop, (propind1,0), bins1, Prop, (propind2,0))
    
#============
#some useful functions for averaging over!
#============
def Size(prev,curr):
    return curr.numSites 

def Prop(prev,curr,propid,measureno=0,usecurr=1):
    #identity function:
    if(usecurr == 1):
        return curr.values[measureno,propid]
    else: 
        if(prev is None):
            return np.nan 
        return prev.values[measureno,propid]

def PropDeltaAvalanche(prev,curr,propid):
    #gets the change in a property during the avalanche (e.g. the stress change, usually negative)
    return  curr.values[1,propid] -  curr.values[0,propid]

def PropNegDeltaAvalanche(prev,curr,propid):
    #gets the negative of a change in a property during the avalanche (e.g. the stress drop, usually positive)
    return  -(curr.values[1,propid] -  curr.values[0,propid])

def PropDeltaLoad(prev,curr,propid):
    #gets change in a property over the loading period
    if(prev is None):
        return np.nan 
    return curr.values[0,propid] - prev.values[1,propid]

def PropNegDeltaLoad(prev,curr,propid):
    #gets change in a property over the loading period
    if(prev is None):
        return np.nan 
    return -(curr.values[0,propid] - prev.values[1,propid])

def clampFunction(prev,curr,clampPropInd,min,max,func,args,measureno = 0):
    val = curr.values[measureno,clampPropInd]
    # print('clamps: ',val,max,min)
    if(val <= max and val >= min):
        return func(prev,curr,*args)
    else:
        return np.nan

def swapFunction(prev,curr,func,args):
    if(prev is None):
        return np.nan
    return func(curr,prev,*args)

from  dklib.running_hist import hist
class histfunc(av_sequencer):
    def __init__(self,func,args,bins):
        self.hist = hist(bins)
        self.func = func
        self.args = args 

    def addAv(self,prev,curr):
        val = self.func(prev,curr,*self.args)
        self.hist.addDat(np.array([val]),0.)
        # binno = np.digitize(self.func(prev,curr,*self.args),self.bins)-1
        # if(binno<0):
        #     #can't rely on index error, as it turns out numpy arrays like to wrap around
        #     return False 
        # try:
        #     self.counts[binno]+=1
        #     return True
        # except IndexError: 
        #     return False  


class histfunc_weighted(av_sequencer):
    def __init__(self,func,args,bins,weightfunc,wargs):
        self.func = func
        self.args = args 
        self.weightfunc = weightfunc 
        self.wargs = wargs 

        self.binN = np.shape(bins)[0]
        self.counts = np.zeros(self.binN-1)
        self.bins = bins 

    def addAv(self,prev,curr):
        val = self.func(prev,curr,*self.args)
        binno = np.digitize(val,self.bins)-1
        # print(self.func1(prev,curr,*self.args1),binno,self.args1)
        if(binno<0):
            return False 
        try:
            fval = self.weightfunc(prev,curr,*self.wargs)
            if(not np.isnan(fval)):
                self.counts[binno]+=fval
            return True
        except IndexError: 
            return False  



import dklib.hist2d
class hist2dfunc(av_sequencer):
    def __init__(self,func1,args1,bins1,func2,args2,bins2):
        self.hist = dklib.hist2d.hist2d(bins1,bins2)
        self.args1 = args1 
        self.func1 = func1
        self.args2 = args2 
        self.func2 = func2
    
    def addAv(self,prev,curr):
        val1 = self.func1(prev,curr,*self.args1)
        val2 = self.func2(prev,curr,*self.args2)
        # print('values: ',val1,val2,np.array(val1),np.array(val2))
        # print(self.func1,self.func2)
        self.hist.addData(np.array([val1]),np.array([val2]))

class histdrop(histfunc):
    def __init__(self,propind,bins):
        super().__init__(PropNegDeltaAvalanche,(propind,),bins)

#what other sequencers are we missing? 
#maybe something that lets us look at prop1_1,prop2_1 then prop1_2,prop2_2 (stress-strain curves with load / unload?)
#not sure that this is an av sequencer?weib(xCenters,1,2.)