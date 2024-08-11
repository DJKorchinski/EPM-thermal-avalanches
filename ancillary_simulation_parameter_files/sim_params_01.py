#simulation params here: 
import numpy as np
import dklib.param_manager as pm


MAX_AVS = 1e6
MAX_TOTAL_STS = 1e6
MAX_STRAIN = 1e2
MAX_SIMULATION_WTIME = 60.*60*(72) #run for up to 72 hours.
MAX_AVALANCHE_GRACE_WTIME = 5.*60. #simulate the final avalanche up to 5 minutes after the max simulation time. 
MAX_AV_SIZE = 1e4

SAVE_OUTPUT_TIMES = np.array([ ]) * 60*60. #don't save the state separately.   
AUTOSAVE_INTERVAL = 0.5*60*60. #output the save file every 30 minutes

HISTOGRAM_STRESS_INTERVAL = 1.0 #make a new px histogram every  HISTOGRAM_STRESS_INTERVAL cumulative stress applied. 
MAP_STRESS_INTERVAL = 5.0 #take 'photo' (O(L^2)) every  MAP_STRESS_INTERVAL cumulative stress applied. 
RECORD_ST_INDS = True
RECORD_STS = True

RECORD_INIT_FAILING_SITES = False

BETA_ARRAY = 1./np.geomspace(1e-5,3e-2,5)
ALPHA_ARRAY = np.array([1.0,2.0])
V_ARRAY = -np.geomspace(1e-30,1e-0,7)
REPLICATE_ARRAY = np.arange(3)
PARAM_MANAGER = pm.param_manager([V_ARRAY,BETA_ARRAY,ALPHA_ARRAY,REPLICATE_ARRAY])
AVALANCHE_WINDOW_IN_TAUS = 3
TAU_R = 1e-6
RENEWAL_K = 2.0
ANNEAL_K = 2.0

#Arrhenius activation related defaults:
DEFAULT_ALPHA_EXPONENT = 2.0
INTER_AVALANCHE_NUMERICAL_TIME = (25-AVALANCHE_WINDOW_IN_TAUS)*TAU_R

def get_params(repno):
    return PARAM_MANAGER.get_params(repno)

#measurement settings:
PX_BINS = np.geomspace(1e-7,1e1,90)
DSIG_BINS = np.append(-np.geomspace(1e1,1e-9),np.geomspace(1e-9,1e1))
AVSIZE_BINS = np.geomspace(1e-3,1e4)
SIG_BINS = np.linspace(-2.,2., 100)