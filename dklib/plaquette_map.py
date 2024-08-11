from dolfin import * 
import numpy as np 
from dklib.mpi_utils import printmpi, is_root
import dklib.arrays
import mpi4py
from numba import jit 

class plaquette_cell_map :
    def __init__(self,mesh,spacing):
        M = mesh
        NL = len(spacing)-1     #plaquettes per side 
        NC = int(M.num_cells()) #number of cells. 
        self.num_plaq = NL**2
        self.num_plaq_per_side = NL
        self.num_cells = NC 
        self.cell_to_plaq = np.zeros(NC,dtype=int)
        self.plaq_to_plaq_xy = np.zeros([self.num_plaq,2],dtype=int)
        self.plaq_xy_to_plaq = np.zeros([NL,NL],dtype=int)-1
        # self.cells_per_plaq = int(NC / self.num_plaq)
        self.cells_local_to_global = np.zeros(NC,dtype=int)
        plaq_cells = dklib.arrays.listoflists(self.num_plaq)#np.zeros([self.num_plaq, self.cells_per_plaq],dtype=int)
        self.cs = []
        self.isroot = is_root()

        plaq_used = 0
        self.plaq_centres = np.zeros([self.num_plaq,2],dtype=float)
        for yi in range(0,NL):
            for xi in range(0,NL):
                self.plaq_xy_to_plaq[xi,yi] = plaq_used
                self.plaq_to_plaq_xy[plaq_used] = [xi,yi]
                self.plaq_centres[plaq_used] = (self.plaq_to_plaq_xy[plaq_used]+0.5) * spacing[-1]/NL
                plaq_used += 1

        for c in cells(M):
            self.cs.append(c)
            x,y,z = c.midpoint().array()
            localind = c.index()
            ind = c.global_index()
            self.cells_local_to_global[localind] = ind
            xi,yi = np.searchsorted(spacing,x)-1,np.searchsorted(spacing,y)-1
            plaq = self.plaq_xy_to_plaq[xi,yi]
            self.cell_to_plaq[localind] = plaq 
            plaq_cells[plaq].append(localind)
            # plaq_cells_assigned[plaq]+=1
        self.plaq_cells = dklib.arrays.ragged(plaq_cells)

        #let's work out how many cells there are per plaq
        #and also what fraction we own?
        self.global_plaq_cell_counts = np.array(self.plaq_cells.lens,copy=True)
        # printmpi('about to all reduce: ',self.global_plaq_cell_counts,type(self.global_plaq_cell_counts))
        mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,self.global_plaq_cell_counts,op=mpi4py.MPI.SUM)
        #okay, we've worked out the global plaq cell counts! So, how much does each cell contribute to the plaquette?
        self.plaq_cellweight = 1./ self.global_plaq_cell_counts
        # printmpi('frac of cells owned: ',self.plaq_cellweight)

#provides a map between vectors from a function spac,and plaquettes.
class plaquette_dof_map:
    def __init__(self,dg0space,cellmap = None,spacing=None):
        M = dg0space.mesh()
        if(cellmap is None):
            self.cmap = plaquette_cell_map(M,spacing)
        else:
            self.cmap = cellmap
        
        dof = dg0space.dofmap()
        self.dofmap = dof
        #the plaq_dofs are ordered by plaquette, and then by cell. So [0,:] = cell0_dof0,cell0_dof1,...cell0,dofN,cell1_dof0... etc
        self.dofs_per_cell = dof.num_entity_dofs(M.geometric_dimension())#this might be a problem later?
        
        # self.dofs_per_plaq = self.cmap.cells_per_plaq*self.dofs_per_cell 
        self.dofs_plaq = np.zeros(self.dofs_per_cell*self.cmap.num_cells,dtype=np.int32)#map from dofs to plaqs
        plaq_dofs_list = dklib.arrays.listoflists(self.cmap.num_plaq)
        for c in self.cmap.cs:
            ci = c.index()
            cdofs = dof.cell_dofs(ci)
            # printmpi('cell: ',ci,c.global_index(),' dofs: ',cdofs)
            plaq = self.cmap.cell_to_plaq[ci]
            plaq_dofs_list[plaq]+=list(cdofs)
            self.dofs_plaq[cdofs] = plaq
        self.plaq_dofs = dklib.arrays.ragged(plaq_dofs_list)
        self.avgcache = np.zeros(self.cmap.num_plaq) #working memory for averaging operation 

    #will return values over all plaquettes
    #offset refers to the dof offset to use (1 for xy, 2 for yx in 2x2 tensors)
    #will automatically iterate in steps of dofstride
    # def plaquette_average(self, dofs, offset = 0, array=None):
    #     if(array is None):
    #         array = np.zeros(self.num_plaq)
    #     for i in range(0,self.num_plaq):
    #         array[i]  = np.sum(dofs[self.plaq_dofs[i,offset::self.dofstride]])
    #     array /= self.cells_per_plaq
    #     return array 

    # def plaquette_average_old(self, dofs,  offset = 0,array=None):
    #     if(array is None):
    #         array = np.zeros(self.num_plaq)
    #     array[:] = np.mean(dofs[self.plaq_dofs[:,offset::self.dofs_per_cell].reshape(self.cmap.num_plaq*self.dofs_per_plaq // self.dofs_per_cell)].reshape(self.cmap.num_plaq,self.dofs_per_plaq//self.dofs_per_cell),1)
    #     return array 

    #must supply an array if rank != 0. Returns None to non-root nodes.
    #collective
    def plaquette_average_mpi(self,dofs,array,offset=0):
        self.avgcache[:] = 0.
        dklib.arrays.sumreduce_strided(dofs,self.dofs_plaq,self.avgcache,offset,self.dofs_per_cell)
        self.avgcache*=self.cmap.plaq_cellweight        
        if(self.cmap.isroot):
            mpi4py.MPI.COMM_WORLD.Reduce(
                [self.avgcache,mpi4py.MPI.DOUBLE],
                [array,mpi4py.MPI.DOUBLE],
                op=mpi4py.MPI.SUM,root=0)
            return array
        else: 
            mpi4py.MPI.COMM_WORLD.Reduce(
                [self.avgcache,mpi4py.MPI.DOUBLE],
                None,
                op=mpi4py.MPI.SUM,root=0)
            array = None 

    
    # def plaquette_average(self, dofs,  offset = 0,array=None):
    #     if(array is None):
    #         array = np.zeros(self.cmap.num_plaq)
    #     dklib.arrays.sumreduce_strided(dofs,self.dofs_plaq,array,offset,self.dofs_per_cell)
    #     return array 
        
    #evaluates the plaquette average, over plaquette ``ind'' 
    #collective operation! Returns same value to all nodes?
    def plaquette_by_ind(self, ind, dofs, offset=0):
        dofInds = np.array(self.plaq_dofs[ind][offset::self.dofs_per_cell],dtype=np.int32) #for some reason this cast is necessary?
        plaq_contrib = np.array([self.cmap.plaq_cellweight[ind]*np.sum( dofs[dofInds] )],dtype=np.float64)
        mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,plaq_contrib,op=mpi4py.MPI.SUM)
        # mpi4py.MPI.COMM_WORLD.Allreduce(plaq_contrib,mpi4py.MPI.IN_PLACE,mpi4py.MPI.SUM)
        #mpi4py.MPI.COMM_WORLD.Allreduce(mpi4py.MPI.IN_PLACE,self.global_plaq_cell_counts,op=mpi4py.MPI.SUM)
        return plaq_contrib[0]


    def assign_plaquette_averages(self,vec,plaquette_averages,offset=0):
        toassign = np.arange(offset,np.size(self.dofs_plaq),self.dofs_per_cell)
        # printmpi('assigning: ',offset,np.size(self.dofs_plaq),self.dofs_per_cell)
        vec[toassign] = plaquette_averages[self.dofs_plaq[toassign]]