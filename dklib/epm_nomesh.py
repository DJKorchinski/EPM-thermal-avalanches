from dolfin import * # pylint: disable=unused-wildcard-import
import dklib.projfunctions
import numpy as np
#from mpi4py import MPI 


#Contains a data structure to describe and methods to evolve a single square of elasto-plastic material.
#Has methods to sample parameters from the EPM, to solve the elastic equations associated with the problem, and maintain thresholds for failure.
class square_nomesh:
    def __init__(self,gridN,r):
        self.thresholds = np.ones(gridN*gridN) # 1.0-self.r.random_sample(self.mesh.num_vertices())
        self.eigStrains = np.zeros(gridN*gridN)
        self.sigs = np.zeros(gridN*gridN)
        self.mechnoise = np.zeros(gridN*gridN)
        self.loadingParam = 0.0
        self.mu = 1.0
        self.N = gridN*gridN
        self.r = r


    def calcsig(self):
        self.sigs = (self.loadingParam - self.eigStrains)*self.mu  + self.mechnoise #mechnoise = the noise from mechanical neighbours 
        
    def addSTZ(self,ind,mag):
        self.eigStrains[ind]+=mag
        #propagate stress:
        #deltanoise = self.r.normal(0.0, 0.25*mag*self.mu,self.N) #could use a different noise source...
        deltanoise = self.r.normal(0.0, 0.25*mag*self.mu / self.N,self.N) #could use a different noise source...
        self.mechnoise+=deltanoise 
        self.mechnoise[ind] -= deltanoise[ind]#don't apply the noise at the location of the failing site. ?
    
    def addSTZAssoc(self,ind,mag):
        self.eigStrains[ind]+=mag * np.sign(self.sigs[ind])
        #propagate stress:
#        deltanoise = self.r.normal(0.0, 0.25*mag*self.mu,self.N) #could use a different noise source...
        deltanoise = self.r.normal(0.0, 0.25*mag*self.mu * (169. / self.N),self.N) #could use a different noise source...
        self.mechnoise+=deltanoise 
        self.mechnoise[ind] -= deltanoise[ind]#don't apply the noise at the location of the failing site. ?

    def getMaxSigOver(self):
        maxSigIndex = np.argmax(self.sigs-self.thresholds)
        return self.sigs[maxSigIndex]-self.thresholds[maxSigIndex],maxSigIndex
    

    def getMaxSigOverAbs(self):
        maxSigIndex = np.argmax(np.abs(self.sigs)-self.thresholds)
        return np.abs(self.sigs[maxSigIndex])-self.thresholds[maxSigIndex],maxSigIndex
    


    #returns sig-sig_y for all sites
    def getX(self):
        return  self.sigs -  self.thresholds


