#=================
#a small helper library to perform on the fly calculations and averaging of avalanche objects. 
#=================

from abc import ABC, abstractmethod
import dklib.avalanche
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
        if(binno<0):
            return False 
        try:
            self.counts[binno]+=1
            self.weights[binno]+= self.func2(prev,curr,*self.args2)
            return True
        except IndexError: 
            return False  
    
    def avgs(self):
        return self.weights / self.counts

#This class can probably be a wrapper for binfunc, using the identity function. Just use to quickly plot, e.g. numSites vs. vsstress1
class binxy(binfunc):
    def __init__(self,prop1,prop2,bins1):
        super().__init__(Prop, (prop1,), bins1, Prop, (prop2,))
    
#============
#some useful functions for averaging over!
#============

def Prop(prev,curr,propname):
    #identity function:
    return getattr(curr,propname) 

def PropDeltaAvalanche(prev,curr,propname):
    #gets the change in a property during the avalanche (e.g. the stress change, usually negative)
    return getattr(curr,propname+'2') - getattr(curr,propname+'1')

def PropNegDeltaAvalanche(prev,curr,propname):
    #gets the negative of a change in a property during the avalanche (e.g. the stress drop, usually positive)
    return getattr(curr,propname+'1')-getattr(curr,propname+'2')

def PropDeltaLoad(prev,curr,propname):
    #gets change in a property over the loading period
    if(prev is None):
        return np.nan 
    return getattr(prev,propname+'2') - getattr(curr,propname+'1')



class histfunc(av_sequencer):
    def __init__(self,func,args,bins):
        self.bins = bins  
        self.counts = np.zeros(len(bins)-1)
        self.func = func
        self.args = args 

    def addAv(self,prev,curr):
        binno = np.digitize(self.func(prev,curr,*self.args),self.bins)-1
        if(binno<0):
            #can't rely on index error, as it turns out numpy arrays like to wrap around
            return False 
        try:
            self.counts[binno]+=1
            return True
        except IndexError: 
            return False  

class histdrop(histfunc):
    def __init__(self,propname,bins):
        super().__init__(PropNegDeltaAvalanche,(propname,),bins)

#what other sequencers are we missing? 
#maybe something that lets us look at prop1_1,prop2_1 then prop1_2,prop2_2 (stress-strain curves with load / unload?)
#not sure that this is an av sequencer?weib(xCenters,1,2.)