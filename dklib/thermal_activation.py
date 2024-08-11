import numpy as np 

def compute_params(lambda1,A,xdot,beta):
    params = {}
    #essential vars
    params['lambda1'] = lambda1
    params['A'] = A
    params['xdot'] = xdot
    params['beta'] = beta 
    #some other cached computations:
    params['lambda2'] = xdot*A*beta
    
    return params
    
    
def params_to_tuple(params):
    return params['lambda1'],params['A'],params['xdot'],params['beta']
    
def thermal_t(x,thermal_params,rng):
    v_inv = 1./(1.-rng.random(size = x.shape))
    lam1,A,xdot,beta = params_to_tuple(thermal_params)
    lam2 = thermal_params['lambda2']
    t = 1. / (lam2) * np.log(1 + np.log(v_inv)*lam2/lam1*np.exp(A*beta*x))
    return t

def compute_next_yield(x,thermal_params,rng):
    lam1,A,xdot,beta = params_to_tuple(thermal_params)
    T_ind = np.argmin(x)
    T = x[T_ind] / xdot
    
    ts = thermal_t(x,thermal_params,rng)
    t_ind = np.argmin(ts)
    tmin = ts[t_ind]
    
    if(T < tmin):
        t = T
        ind = T_ind
        thermal_yield = False
    else:
        t = tmin
        ind = t_ind
        thermal_yield = True 
    
    return ind,t,thermal_yield #returns the site index, yielding time, and yield type (Thermal or mechanical)