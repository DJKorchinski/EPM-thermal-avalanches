import numpy as np
import scipy.special
import scipy.integrate
import numba 

def tfailLin(us,sigma,sigma_th,p,s,debug_level = -1):
    #s = sign of failure. + is the usual positive direction (sigma = sigma_th). - is the negative direction (sigma = -sigma_th) 
    v = p.v
    beta = p.beta 
    lmbda = p.lmbda 
    
    xs =sigma_th - s* sigma 
    sv = s*v 
    #compute the loading time:
    log_term_1 =  np.exp(-beta * xs)
    log_term_2 = sv * beta / lmbda * np.log(us)
    Ts = (xs) / (-sv) + 1/(-sv*beta) * np.log( log_term_1 + log_term_2  )

    #for large values, the exponential dominates, and we can use the following to avoid catastrophic cancellation: 
    use_lin_approx = log_term_1 / np.abs(log_term_2) > 1e10
    # use_lin_approx = log_term_1 / np.abs(log_term_2) > 1e10 ## I'm restricting to using the case that log_term_2 > 0, because if log_term_2<0, then we get negative times.
    Ts[use_lin_approx] = -np.log(us[use_lin_approx])/ (lmbda * np.exp(-xs[use_lin_approx]*beta))

    #evaluating log(1+x) is prone to error for small x. In that case, use log(1+x) = log1p(x)
    #similarly, expm1(x) = exp(x)-1 helps avoid error. 
    off1_arg = (np.expm1(-beta * xs)) + log_term_2
    use_off1 = np.abs(off1_arg) < 1e-9
    Ts[use_off1] = (xs[use_off1]) / (-sv) + 1/(-sv*beta) * np.log1p( off1_arg[use_off1]  )


    Ts[np.isnan(Ts)] = np.inf
    if(s > 0):
        #if we are looking at positive failure directions, check to see if x = 0 occurs. 
        Ts = np.minimum( np.maximum(0,(sigma_th - s*sigma)/-p.v),Ts )
    return Ts

def tfailQuad(us,sigma,sigma_th,p,s,debug_level = -1):
    #s = sign of failure. + is the usual positive direction (sigma = sigma_th). - is the negative direction (sigma = -sigma_th)
    v = p.v
    beta = p.beta 
    lmbda = p.lmbda 
    
    xs = sigma_th - s* sigma 
    sv = s*v 

    delta =  -2*(-sv) * np.sqrt(beta) * np.log(us) / (np.sqrt(np.pi) * lmbda) 
    Ts = (xs) / (-sv) + 1/(sv*np.sqrt(beta)) * scipy.special.erfinv(scipy.special.erf(np.sqrt(beta) * xs) - delta) 

    #now, we're going to check if erfc_betax + delta is small. 
    erfc_betax = scipy.special.erfc(np.sqrt(beta) * xs)
    gamma =  erfc_betax+delta
    use_erfc_filter = np.logical_and(gamma < 1e-9,gamma > 0) #basically, erfinv(1-gamma) = erfcinv(gamma), RHS is more accurate for small gamma. 
    Ts[use_erfc_filter] = xs[use_erfc_filter]/(-sv) + 1/(sv*np.sqrt(beta)) * scipy.special.erfcinv( gamma[use_erfc_filter] )

    #okay, what if np.sqrt(beta)*xs >> delta? In the worst case, delta doesn't affect the value at all. We should then use  
    use_first_order_in_delta = np.logical_and( np.logical_or( np.abs(delta / np.sqrt(beta) * xs)<1e-10, np.abs(delta / erfc_betax) < 1e-10) , np.abs(np.exp(beta*xs**2) * delta) < 1 )
    Ts[use_first_order_in_delta] = 0.5*np.exp(beta*(xs[use_first_order_in_delta])**2)*delta[use_first_order_in_delta] * np.sqrt(np.pi) / ((-sv)*np.sqrt(beta))

    if(debug_level > 1):
        print('number of nans: ',np.sum(np.isnan(Ts)))
    Ts[np.isnan(Ts)] = np.inf
    if(s > 0):
        #if we are looking at positive failure directions, check to see if x = 0 occurs. 
        Ts = np.minimum( np.maximum(0,(sigma_th - s*sigma)/-p.v),Ts )
    
    if(debug_level > 1):
        tot = np.sum(Ts<0)
        if(tot > 0):
            print('number of negative Ts',tot)
            badind = np.argmin(Ts)
            T1 =  (xs) / (-sv) + 1/(sv*np.sqrt(beta)) * scipy.special.erfinv(scipy.special.erf(np.sqrt(beta) * xs) - delta) 
            T2 = 0.5*np.exp(beta*(xs)**2)*delta * np.sqrt(np.pi) / ((-sv)*np.sqrt(beta))
            T3 =   xs/(-sv) + 1/(sv*np.sqrt(beta)) * scipy.special.erfcinv( gamma )
            use_first_order_in_delta2 = np.logical_and( np.logical_or( np.abs(delta / np.sqrt(beta) * xs)<1e-10, np.abs(delta / erfc_betax) < 1e-10) , np.abs(np.exp(beta*xs**2) * delta) < 1 )



            import pdb; pdb.set_trace()
            

    return Ts    


#this function loads the system, between avalanches. 
#begins with numerical loading for a time period tnumerical.
# @profile
def load_to_fail_inter_avalanche(mf,rng,tnumerical=1e-6*25,tmax=1e200,debug_level = -1):
    #first, apply numerical loading for tnumerical:
    if(tnumerical>0):
        sites,time = load_to_fail_numerical(mf,rng,tnumerical,debug_level)
        if(debug_level > 0):
            if(sites.size >0):
                print('finished inter_avalanche using numerical',sites,time)
        if(sites.size > 0):
            return sites,time
    
    if(mf.params.v == 0):
        return load_to_fail_static_analytical(mf,rng,tmax,debug_level)

    #now, we try to accelerate the inter_avalanche period using the analytical form:
    alpha = mf.params.alpha
    if(alpha == 1):
        loadfunc = tfailLin
    elif(alpha==2):
        loadfunc = tfailQuad
    else:
        #if we don't have an analytical form for alpha, then let's just resort to numerical integration.
        return load_to_fail_numerical(mf,rng,tmax,debug_level)
    
    #if we made it here, we've set the loadfunc, and need to generate some us. 
    uplus = 1-rng.uniform(size=mf.N)
    uminus = 1-rng.uniform(size=mf.N)
    return load_to_fail_analytical(mf,uplus,uminus,loadfunc,tmax,debug_level)



#this function loads the system, using an analytical form for tfail.
#it makes the assumption that \Sigma_r is zero at all sites, and that all sites have exceeded the tau_last barrier. 
# @profile
def load_to_fail_analytical(mf,uplus,uminus,tFailFunc,tmax=1e200,debug_level = -1):
    Tplus = tFailFunc(uplus,mf.sigma,mf.sigma_th,mf.params,1,debug_level)
    Tminus = tFailFunc(uminus,mf.sigma,mf.sigma_th,mf.params,-1,debug_level) 
    return choose_and_do_loading(mf,Tplus,Tminus,tmax)


def choose_and_do_loading(mf,Tplus, Tminus,tmax):
    #find the smallest time to failure:
    indplus = np.argmin(Tplus)
    indminus = np.argmin(Tminus)
    tplusmin = Tplus[indplus]
    tminusmin = Tminus[indminus]

    #choose the failing index from positive and negative directions. 
    positiveFail = (tplusmin  < tminusmin) 
    if(positiveFail):
        tmin = tplusmin 
        ind_array = [indplus] 
    else:
        tmin = tminusmin
        ind_array = [indminus]
    
    #if the failure time is greater than the maximum time to yielding, then: 
    if(tmin > tmax):
        tmin = tmax
        ind_array = []

    #advance the clock: 
    mf.advance_time(tmin)
    # mf.t+=tmin

    #computing the residucal stress decay at each site, according to its dt.
    stress_decay = np.exp(-tmin / mf.params.tau) 
    mf.sigma += tmin * -mf.params.v + mf.sigma_res*(1-stress_decay)
    mf.sigma_res *= stress_decay 

    mf.compute_x()

    #check for other failing sites? This (very unlikely event) could occur if sigma_r was very small, but big enough to push a site over to failure?
    xfail_sites = np.arange(0,mf.N)[mf.x<0]
    for ind in xfail_sites:
        if(not ind in ind_array):
            ind_array.append(ind)
    return np.array(ind_array),tmin 

def calculate_signed_rates(mf,sign):
    beta = mf.params.beta 
    alpha = mf.params.alpha
    return (mf.params.lmbda) * np.exp( -( np.power(np.maximum(0,(mf.sigma_th - sign*mf.sigma)),alpha) * beta )  )


#loads the system to failure with zero driving
def load_to_fail_static_analytical(mf,rng,tmax,debug_level = -1):
    upos = 1-rng.uniform(size=mf.N)
    uneg = 1-rng.uniform(size=mf.N)
    lambda_pos = calculate_signed_rates(mf,+1.0)
    lambda_neg = calculate_signed_rates(mf,-1.0)
    tfail_pos = np.log(upos)/-lambda_pos
    tfail_neg = np.log(uneg)/-lambda_neg
    return choose_and_do_loading(mf,tfail_pos,tfail_neg,tmax)
    
#===========
#the exact, numerical integration approach.
#===========
#calculates the Arrhenius rate: 
@numba.njit(cache=True)
def calc_rates_inst_single(sigma,sigma_th,lmbda,beta,s,alpha):
    return lmbda*np.exp(-beta*(sigma_th- s * sigma)**alpha)

@numba.njit(cache=True)
def calc_rates_inst_array(sigma,sigma_th,lmbda,beta,s,alpha):
    return lmbda*np.exp(-beta*(sigma_th- s * sigma)**alpha)

# def calc_sigma_deferred(dt,sigma,sigma_th,sigma_r,s,p,alpha):
#     return sigma-p.v*dt+sigma_r*(1-np.exp(-dt/p.tau))


@numba.njit(cache=True)
def calc_sigma_deferred_single(dt,sigma,sigma_r,v,tau):
    return sigma - v * dt + sigma_r*(1-np.exp(-dt/tau))


@numba.njit(cache=True)
def calc_sigma_deferred_array(dt,sigma,sigma_r,v,tau):
    return sigma - v * dt + sigma_r*(1-np.exp(-dt/tau))

# def calc_rates_deferred(dt,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha):
#     return calc_rates_inst(calc_sigma_deferred(dt,sigma,sigma_th,sigma_r,s,p,alpha), sigma_th,p,s,alpha)


@numba.njit(cache=True)
def calc_rates_deferred_single(dt,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha):
    deferred_sigma = calc_sigma_deferred_single(dt,sigma,sigma_r,v,tau)
    return calc_rates_inst_single(deferred_sigma,sigma_th,lmbda,beta,s,alpha)

@numba.njit(cache=True)
def calc_rates_deferred_array(dt,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha):
    deferred_sigma = calc_sigma_deferred_array(dt,sigma,sigma_r,v,tau)
    return calc_rates_inst_array(deferred_sigma,sigma_th,lmbda,beta,s,alpha)


@numba.njit(cache=True)
def calc_rates_du_t(dt,u,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha):
    return (1-u)*calc_rates_deferred_single(dt,sigma, sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha)

@numba.njit(cache=True)
def calc_x_deferred(t,u_t,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha):
    return sigma_th - s * calc_sigma_deferred_single(t,sigma,sigma_r,v,tau)

#calculates the time to failure for a particular site, up to tmax. 
def time_to_fail_numerical(u,tstart,tmax,sigma,sigma_th,sigma_r,p,s,alpha,debug_level=-1):
    solver_event = lambda t,u_t,sigma,sigma_th,sigma_r,v,lmbda,beta,tau,s,alpha : u_t - u
    x_fail_event = calc_x_deferred

    solver_event.terminal = True
    x_fail_event.terminal=True
    max_step_size = (tmax-tstart)*min(5-1,1e-1 / max(-p.v,1e-90) )
    integr = scipy.integrate.solve_ivp(calc_rates_du_t,(tstart,tmax),y0 = np.array([0.0]),\
                                        events=[solver_event,x_fail_event],\
                                       args=(sigma,sigma_th,sigma_r,p.v,p.lmbda,p.beta,p.tau,s,alpha),\
                                      max_step = max_step_size,atol=1e-14,rtol = 1e-12)
    if(debug_level > 1):
        print(integr)
    

    if(np.size(integr.t_events) > 0):
        if(integr.t_events[0].size > 0):
            return integr.t_events[0][0]
        else: 
            return integr.t_events[1][0]
    else:
        return np.inf

def determine_sites_to_calculate_numerical(mf,uplus,uminus,tstarts,sites_immediately_failing,tfail,debug_level = -1):
    p = mf.params
    alpha = p.alpha

    #now, we should check whether it's possible for sites to fail (i.e. if their rates are low enough to avoid failure)
    max_rates_pos = np.maximum(
        calc_rates_deferred_array(tstarts,mf.sigma,mf.sigma_th,mf.sigma_res,p.v,p.lmbda,p.beta,p.tau,1,alpha),\
        calc_rates_deferred_array(tfail,mf.sigma,mf.sigma_th,mf.sigma_res,p.v,p.lmbda,p.beta,p.tau,1,alpha))
        
    max_rates_neg = np.maximum(
        calc_rates_deferred_array(tstarts,mf.sigma,mf.sigma_th,mf.sigma_res,p.v,p.lmbda,p.beta,p.tau,-1,alpha),\
        calc_rates_deferred_array(tfail,mf.sigma,mf.sigma_th,mf.sigma_res,p.v,p.lmbda,p.beta,p.tau,-1,alpha))

    cannot_fail_prob_pos = max_rates_pos * (tfail - tstarts) < -np.log(1-uplus)
    cannot_fail_prob_neg = max_rates_neg * (tfail - tstarts) < -np.log(1-uminus)

    res_stress_gain = mf.sigma_res * (1 - np.exp(-tfail / p.tau))
    could_hit_zero_pos = np.logical_or(+1*(mf.sigma - p.v * tfail + res_stress_gain)  > mf.sigma_th  , 1*(mf.sigma + res_stress_gain) > mf.sigma_th)
    could_hit_zero_neg = np.logical_or(-1*(mf.sigma - p.v * tfail + res_stress_gain)  > mf.sigma_th  , -1*(mf.sigma + res_stress_gain) > mf.sigma_th)
    cannot_fail_prob_pos = np.logical_and(cannot_fail_prob_pos,np.logical_not(could_hit_zero_pos))
    cannot_fail_prob_neg = np.logical_and(cannot_fail_prob_neg,np.logical_not(could_hit_zero_neg))

    
    sites_cannot_fail_prob = np.logical_and(cannot_fail_prob_pos,cannot_fail_prob_neg)
    sites_to_compute = np.logical_and(np.logical_not(sites_cannot_fail_prob),np.logical_not(sites_immediately_failing))
    return sites_to_compute

# @profile
def load_to_fail_numerical(mf,rng,tmax=1e99,debug_level = -1):
    #now, use the ode_solver method. 
    p = mf.params
    alpha = p.alpha
    uplus = rng.uniform(size=mf.sigma.size)
    uminus = rng.uniform(size=mf.sigma.size)

    #compute the legal start times for various sites to fail: 
    tstarts = mf.tsolid


    #find the failing times.:
    tfail = tmax
    fail_ind = -1
    Tplus = np.ones(mf.sigma.shape)
    Tminus = np.ones(mf.sigma.shape)


    #let's check for sites that are immediately going to fail.
    sigma_at_tstart = calc_sigma_deferred_array(tstarts,mf.sigma,mf.sigma_res,p.v,p.tau)
    sites_immediately_failing = np.abs(sigma_at_tstart) > mf.sigma_th
    Tplus[sites_immediately_failing] = tstarts[sites_immediately_failing]
    Tminus[sites_immediately_failing] = np.inf 

    #and now if we have some sites that are immediately going to fail, we should see which among them is the first to fail.
    num_sites_immediately_failing = np.sum(sites_immediately_failing)
    if(num_sites_immediately_failing > 0):
        index_among_immediately_failing_sites = np.argmin(tstarts[sites_immediately_failing])
        immediately_failing_site_indices = np.arange(mf.N)[sites_immediately_failing]
        fail_ind = immediately_failing_site_indices[index_among_immediately_failing_sites]
        tfail = tstarts[fail_ind]

    sites_to_compute = determine_sites_to_calculate_numerical(mf,uplus,uminus,tstarts,sites_immediately_failing,tfail,debug_level)

    #now, sort the sites by failure proximity: 
    failure_proximity = np.zeros((mf.N,2))
    failure_proximity[:,0] = np.arange(mf.N)
    failure_proximity[:,1] = np.minimum( mf.sigma_th - mf.sigma-np.maximum(mf.sigma_res,0),mf.sigma_th + mf.sigma + np.minimum(mf.sigma_res,0) )
    failure_proximity = failure_proximity[sites_to_compute,:]
    sorted_list = sorted(failure_proximity,key = lambda pair : pair[1])

    # print('sorted list length:',len(sorted_list))
    # import pdb; pdb.set_trace()

    for j,pair in enumerate(sorted_list):
        ind = int(pair[0])
        sig = mf.sigma[ind]
        sig_th = mf.sigma_th[ind]
        sig_r = mf.sigma_res[ind]
        start_time = tstarts[ind]
        if(start_time > tfail or not sites_to_compute[ind]):
            Tplus[ind] = np.inf 
            Tminus[ind] = np.inf 
            continue

        Tplus[ind] =  time_to_fail_numerical(uplus[ind],start_time,tfail,sig,sig_th,sig_r,p,+1,alpha,debug_level)
        if(debug_level>1):
            print(ind,'arguments:',(uplus[ind],start_time,tfail,sig,sig_th,sig_r,p,+1,alpha),'tplus:',Tplus[ind])
            print(Tplus[ind],tfail,Tplus[ind]<tfail)
        updated_tfail = False 
        if(Tplus[ind] < tfail):
            fail_ind = ind 
            tfail = Tplus[ind]
            updated_tfail = True 
        Tminus[ind] = time_to_fail_numerical(uminus[ind],start_time,tfail,sig,sig_th,sig_r,p,-1,alpha,debug_level)
        if(Tminus[ind] < tfail):
            fail_ind = ind 
            tfail = Tminus[ind]
            updated_tfail = True 
        if(updated_tfail and j < len(sorted_list) - 1): #i.e. if there is at least one more site to re-compute.
            sites_to_compute = determine_sites_to_calculate_numerical(mf,uplus,uminus,tstarts,sites_immediately_failing,tfail,debug_level)
            # print('recompute length: ',np.sum(sites_to_compute))
    if(debug_level>1):
        print('finished numerical loop:',fail_ind,tfail)
    #if the failure time is greater than the maximum time to yielding, then: 
    if(tfail > tmax):
        tmin = tmax
        ind_array = []
    elif(fail_ind == -1):
        tmin = tmax
        ind_array = []
    else:
        tmin = tfail
        ind_array = [fail_ind]
    

    #advance the clock: 
    mf.advance_time(tmin)
    # mf.t+=tmin 

    #computing the residucal stress decay at each site, according to its dt.
    stress_decay = np.exp(-tmin / mf.params.tau) 
    mf.sigma += tmin * -mf.params.v + mf.sigma_res*(1-stress_decay)
    mf.sigma_res *= stress_decay 

    mf.compute_x()

    if(debug_level > 2):
        print('numerical_output: ',ind_array,tmin)
        if(tmin == 0.0):
            import pdb; pdb.set_trace()
    return np.array(ind_array),tmin