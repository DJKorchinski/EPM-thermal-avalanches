#simulation params here: 
import numpy as np
import dklib.param_manager as pm


MAX_AVS = 3e6 
MAX_TOTAL_STS = 1e7
MAX_STRAIN = 1e2
# MAX_SIMULATION_WTIME = 60.*60*(24*7+12) #run for up to 72 hours.
MAX_SIMULATION_WTIME_L = {32:  3.5, 64: 3.5, 128: 5.5, 256:7.5, 512: 7.5 }# in days
MAX_SIMULATION_WTIME_L = { key : MAX_SIMULATION_WTIME_L[key]*3600*24 for key in MAX_SIMULATION_WTIME_L.keys() } #conversion to seconds.
MAX_AVALANCHE_GRACE_WTIME = 15.*60. #simulate the final avalanche up to 5 minutes after the max simulation time. 
MAX_AV_SIZE = 1e4

SAVE_OUTPUT_TIMES = np.array([ ]) * 60*60. #don't save the state separately.   
AUTOSAVE_INTERVAL = 0.5*60*60. #output the save file every 30 minutes

HISTOGRAM_STRESS_INTERVAL = 1.0 #make a new px histogram every  HISTOGRAM_STRESS_INTERVAL cumulative stress applied. 
MAP_STRESS_INTERVAL = 5.0 #take 'photo' (O(L^2)) every  MAP_STRESS_INTERVAL cumulative stress applied. 
RECORD_ST_INDS = False
RECORD_STS = True

RECORD_INIT_FAILING_SITES = True

# BETA_ARRAY = 1./np.array([1e-9]) #these are to fill in the temperature gaps in our existing data.
ALPHA_ARRAY = np.array([1.0,2.0])
V_ARRAY = -np.array([1e-4,1e-2,1e0,1e2])
T_INDEX_ARRAY=np.arange(12)
PARAM_MANAGER = pm.param_manager([V_ARRAY,ALPHA_ARRAY,T_INDEX_ARRAY])
AVALANCHE_WINDOW_IN_TAUS = 3
TAU_R = 1e-6
RENEWAL_K = 2.0
ANNEAL_K = 2.0

#Arrhenius activation related defaults:
DEFAULT_ALPHA_EXPONENT = 2.0
INTER_AVALANCHE_NUMERICAL_TIME = (25-AVALANCHE_WINDOW_IN_TAUS)*TAU_R



def get_params(repno,L):
    v,alpha,T_index = PARAM_MANAGER.get_params(repno)

    #scaling parameters:
    theta = 0.57
    d = 2
    Tcross = lambda L,alpha : L**((-d*alpha)/(theta+1))
    T_gdot_cross2 = lambda gdot_tau,alpha : gdot_tau**(alpha)
    
    gdot = -v*TAU_R
    Tmax = 10*Tcross(L,alpha)
    Tmin = 1e-2 * T_gdot_cross2(gdot,alpha)
    Ts = np.geomspace(Tmin,Tmax,T_INDEX_ARRAY.size)
    return (v,1./Ts[T_index],alpha)
    

#measurement settings:
PX_BINS = np.geomspace(1e-9,1e1,60)
DSIG_BINS = np.append(-np.geomspace(1e1,1e-9),np.geomspace(1e-9,1e1))
AVSIZE_BINS = np.geomspace(1e-3,1e4)
SIG_BINS = np.linspace(-2.,2., 100)