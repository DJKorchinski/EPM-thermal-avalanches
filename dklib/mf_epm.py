import numpy as np 

#describes brownian walkers, with a diffusion constant D^2/2. 
class ThermalParams():
    def __init__(self,D = 0.001, v = -0.007, tau = 1e-6):
        self.D = D
        self.v = v 
        self.tau = tau 

#parameters for describing arrhenius activation rates. 
class ArrheniusThermalParams():
    def __init__(self,v=-1e-3,beta=1e-8,alpha = 2.0,tau=1e-6):
        self.tau = tau 
        self.beta = beta 
        self.v = v 
        self.lmbda = 1/tau
        self.alpha = alpha

class MFEPM():
    def __init__(self,N,params):
        self.x = np.zeros(N)
        self.sigma = np.zeros(N)
        self.sigma_th = np.zeros(N)
        self.sigma_res = np.zeros(N) #residual stress, with exponential decay. 
        self.tlast = -1e99*np.ones(N) #last failure time. 
        self.tsolid = np.zeros(N) #the time remaining until a site is solid. Should be zero or positive!

        #
        self.dsig_temp = np.zeros(N)

        self.cumulative_eigensetrain = np.zeros(N)
        
        #accumulated time.
        self.t = 0.0  
        self.N = N 
        self.params = params 

    def compute_x(self):
        np.abs(self.sigma, out = self.x)
        np.negative(self.x,out = self.x)
        np.add(self.sigma_th,self.x,out=self.x)

    def fluidization_time(self):
        return 2.0 * self.params.tau
    
    def solid_site_inds(self):
        return self.tsolid==0
        # return (self.t - self.tlast) > self.fluidization_time() 
    
    def advance_time(self,dt):
        self.t+=dt 
        self.tsolid = np.maximum(0.0,self.tsolid-dt)

    dsig_temp = None 


    def apply_kicks(self,failingSites,kickSource,rng,renewal_k = 2., constant_stress = False, cumulative_eigenstrain=None):
        from dklib.mpi_utils import is_root
        mf = self
        dsig_temp = self.dsig_temp 
        for ind in failingSites:
            kickSource.compute_kicks(ind,rng,dsig_temp)
            
            if(is_root()):
                mf.sigma_th[ind] = rng.weibull(renewal_k,)
                sig = mf.sigma[ind]
                #it's important we count the eigenstrain before applying the 'constant_stress' correction, since the -np.mean(dsig_temp) loading is external. 
                if(not( cumulative_eigenstrain  is None)):
                    cumulative_eigenstrain[ind] += -sig*dsig_temp[ind]

                if(constant_stress):
                    dsig_temp[:] -= np.mean(dsig_temp)
                mf.sigma_res += sig * dsig_temp
        if(is_root()):
            mf.tlast[failingSites] = mf.t
            mf.tsolid[failingSites] = mf.fluidization_time()
            mf.compute_x()
