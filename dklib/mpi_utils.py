from mpi4py import MPI
from sys import stdout 
from time import sleep 
import numpy as np 

def get_rank():
    return MPI.COMM_WORLD.Get_rank()
def is_root():
    return get_rank()==0

def printmpi(*msgs):
    print('thread no: %d / %d'%(get_rank(),MPI.COMM_WORLD.Get_size()),*msgs)
    stdout.flush()

def printroot(*msgs):
    if(get_rank() == 0):
        printmpi(*msgs)

def delaympi(delay=0.1):
    sleep(.1*MPI.COMM_WORLD.Get_rank())



            # stoppingForSave=np.array([False],dtype=np.bool8)
            # if(is_root() and (np.mod(avNum-lastAvSave,1000)==0 or elapsedHours-lastProgSaveHours> 0.5) ):
            #     stoppingForSave[0] = True
            # tMPI.COMM_WORLD.Bcast(stoppingForSave,root=0)
def unifyvar(var,dtype=np.float64,mpitype = None):
    arr = np.array([var],dtype=dtype)
    if(is_root()):
        arr[0] = var
    if(mpitype is None):
        MPI.COMM_WORLD.Bcast(arr,root=0)
    else:
        MPI.COMM_WORLD.Bcast((arr,mpitype),root=0)
    return arr[0]

def unifyarr(arr,dtype=np.float64,mpitype = None):
    arrsize = 0 
    if(is_root()):
        arr_np = np.array(arr,dtype = dtype)
        arrsize = arr_np.size 
    arrsize = unifyvar(arrsize,dtype=np.int64)
    if(not is_root()):
        arr_np = np.zeros(arrsize, dtype=dtype)

    if(mpitype is None):
        MPI.COMM_WORLD.Bcast(arr_np,root=0)
    else:
        MPI.COMM_WORLD.Bcast((arr_np,mpitype),root=0)
    return arr_np

