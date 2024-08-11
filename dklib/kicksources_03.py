import dklib.epm_plaquette as epm
import numpy as np 
from dklib.mpi_utils import is_root
class shuffled_kernel:
    def __init__(self,L):
        self.L = L
        sq = epm.square(L,'simpleShearDisplacement')
        sq.solve_u_new_bc()
        plaq_id = sq.cmap.plaq_xy_to_plaq[L//2,L//2] #apply the kick to the center 
        sq.addSTZ(plaq_id)
        sq.solve_u_const_bc()
        sq.calc_sig_xy()
        sq.calc_sig_xy_plaqs()
        dsig_temp = np.zeros(L*L)
        sq.get_sig_xy_plaqs(dsig_temp)
        self.centre = dsig_temp[plaq_id] 
        self.kern = np.array(list(dsig_temp[:plaq_id]) + list(dsig_temp[plaq_id+1:]))
        self.kern_shuffled = np.zeros(shape = self.kern.shape)



    def compute_kicks(self,failing_ind, rng,dsig_temp):
        if(is_root()):
            self.kern_shuffled[:] = self.kern[:]
            dsig_temp[failing_ind] = self.centre 
            rng.shuffle(self.kern_shuffled)
            dsig_temp[:failing_ind] = self.kern_shuffled[:failing_ind]
            dsig_temp[failing_ind+1:] = self.kern_shuffled[failing_ind:]


class fem_kernel:
    def __init__(self,L):
        self.L = L 
        sq = epm.square(L,'simpleShearDisplacement')
        sq.solve_u_new_bc()
        self.sq = sq 

    def compute_kicks(self,failing_ind,rng,dsig_temp):
        sq = self.sq 
        sq.eigenstrain.vector()[:] = 0.
        sq.addSTZ(failing_ind,1.)       
        sq.solve_u_const_bc()
        sq.calc_sig_xy()
        sq.calc_sig_xy_plaqs()
        sq.get_sig_xy_plaqs(dsig_temp)
        
        # return self.dsig
        
class gaussian_kernel:
    def __init__(self,L,A=1.0):
        self.L = L
        self.A = A
    
    def compute_kicks(self,failing_ind,rng,dsig_temp):
        dsig_temp[:] = rng.normal(0.0,self.A / self.L**(1.),size = dsig_temp.size)
        dsig_temp[failing_ind] = -1.0