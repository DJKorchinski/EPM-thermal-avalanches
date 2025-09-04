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


        self.net_fails = np.zeros(N,dtype=np.int64) #counts how many times each site has failed, in each direction. 
        self.sigma_th_wells = [ {} for _ in range(N) ]

        #
        self.dsig_temp = np.zeros(N)

        self.cumulative_eigenstrain = np.zeros(N)
        
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
        total_stress_response = 0
        for ind in failingSites:
            kickSource.compute_kicks(ind,rng,dsig_temp)
            
            if(is_root()):
                # To update the sigma_th, we need to know which well the site is now in.
                failure_sign = np.sign(mf.sigma[ind])
                mf.net_fails[ind] += failure_sign #record the direction of the failure.
                net_failure = mf.net_fails[ind]
                # If this well has not been visited before, we need to assign it a new threshold.
                if( not (net_failure in mf.sigma_th_wells[ind].keys() )):
                    new_threshold = rng.weibull(renewal_k,)
                    mf.sigma_th_wells[ind][net_failure] = new_threshold
                # Finally, we set the threshold to the value of the current well.
                mf.sigma_th[ind] = mf.sigma_th_wells[ind][net_failure]

                sig = mf.sigma[ind]
                dsig_temp *= sig
                #it's important we count the eigenstrain before applying the 'constant_stress' correction, since the -np.mean(dsig_temp) loading is external. 
                if(not( cumulative_eigenstrain  is None)):
                    cumulative_eigenstrain[ind] += -dsig_temp[ind]
                mf.cumulative_eigenstrain[ind] += -dsig_temp[ind]
                stress_response = np.mean(dsig_temp)
                if(constant_stress):
                    dsig_temp[:] -= stress_response
                mf.sigma_res += dsig_temp
                total_stress_response+=stress_response
        if(is_root()):
            mf.tlast[failingSites] = mf.t
            mf.tsolid[failingSites] = mf.fluidization_time()
            mf.compute_x()
        return total_stress_response 