#simulation params here: 
import numpy as np

MAX_AVS = 1e6
MAX_STRAIN = 60.0
MAX_SIMULATION_WTIME = 120.0*3600.0 #run for 120 hours (five days). 
MAX_AV_SIZE = 1e4
AVALANCHE_WINDOW_IN_TAUS = 3

DATA_FOLDER = 'data_08/'
D_ARRAY = np.array([1e-4,1e-5,1e-6])
L_ARRAY = np.array([32,64,128,256])
V_ARRAY = np.array([-1e1,-3e-0,-1e-0,-6e-1,-3e-1,-1e-1])
N_REP = D_ARRAY.size * L_ARRAY.size * V_ARRAY.size
TAU_R = 1e-6

def get_params(repno):
    D = D_ARRAY[repno%D_ARRAY.size]
    v = V_ARRAY[(repno//D_ARRAY.size ) // L_ARRAY.size]
    return D,v 

def L_ind(repno):
    return (repno//D_ARRAY.size)%L_ARRAY.size

def L(repno):
    return L_ARRAY[L_ind(repno)]


PX_BINS = np.geomspace(1e-9,1e1,60)