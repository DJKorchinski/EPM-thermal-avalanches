from dolfin import * # pylint: disable=unused-wildcard-import
# import dklib.projfunctions
import dklib.plaquette_map
import numpy as np
import numpy.linalg
import mpi4py
from dklib.mpi_utils import printroot,printmpi
#import scipy as sp 
#import scipy.linalg


#Contains a data structure to describe and methods to evolve a single square of elasto-plastic material.
#Has methods to sample parameters from the EPM, to solve the elastic equations associated with the problem, and maintain thresholds for failure.
class square:
    def __init__(self,gridN,loadingCondition):
        self.gridN = gridN
        self.wid = gridN 
        self.mesh = RectangleMesh(MPI.comm_world,Point(0.,0.), Point(gridN,gridN),gridN,gridN,"crossed")
#        self.mesh = RectangleMesh(Point(0.,0.), Point(wid,wid),gridN,gridN,"crossed")
        self.dim = self.mesh.geometry().dim()
        self.mu, self.lmbda = Constant(1.), Constant(1.)
        self.U = FunctionSpace(self.mesh,'DG',0)
        self.W = TensorFunctionSpace(self.mesh,'DG',0,shape=(self.dim,self.dim))#,symmetry=True) 
        self.V = VectorFunctionSpace(self.mesh,'Lagrange',1)
        self.eps = Function(self.W,name="Strain") 
        self.sig = Function(self.W,name="Stress")
        # self.sigdev = Function(self.W,name="Deviatoric Stress")
        self.eigenstrain = interpolate(Constant(((0,0),(0,0))),self.W)
        # self.equivStress = interpolate(Constant(0),self.U)
        # self.thresholds = np.ones(self.mesh.num_vertices()) # 1.0-self.r.random_sample(self.mesh.num_vertices())

        #initializing the inclusion expression:
        phi = np.pi / 4.0
        self.inclusionExpression = Expression((("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*(2*nhatx*nhatx - 1):0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*2.0*nhatx*nhaty:0.0"),("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*2.0*nhatx*nhaty:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*(2*nhaty*nhaty - 1):0.0")),degree=5,x0c=0,x1c=0,radsq=.25,nhatx=cos(phi),nhaty=sin(phi),strainmag = 1.0)
        #set a raw (sigxx, sigxy, sigyy) type matrix expression:
        self.inclusionExpression2 = Expression((("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxx:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxy:0.0"),("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxy:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigyy:0.0")),degree=5,x0c=0,x1c=0,radsq=.25,sigxx = 0.,sigxy =1.,sigyy=0.,strainmag = 1.0)

        if(loadingCondition != None):
            self.setLoadingCondition(loadingCondition)
        

        #build the forms for assembly:
        self.init_forms()
        self.cmap = dklib.plaquette_map.plaquette_cell_map(self.mesh, np.linspace(0,gridN,gridN+1) )
        self.Umap = dklib.plaquette_map.plaquette_dof_map(self.U,cellmap=self.cmap)
        self.Wmap = dklib.plaquette_map.plaquette_dof_map(self.W,cellmap=self.cmap)
        
        #setting the thresholds and the x values: 
        self.plaq_thresh = np.ones(self.cmap.num_plaq) #probably want to pass some kind of initiation / renewal class
        self.plaq_x      = np.ones(self.cmap.num_plaq) 
        self.plaq_sig_xy = np.zeros(self.cmap.num_plaq)
        self.plaq_eps_xy = np.zeros(self.cmap.num_plaq)
        if(not self.cmap.isroot):
            self.plaq_x      = None
            self.plaq_sig_xy = None 
    
    # def 

    def init_forms(self):
        self.u = Function(self.V)
        self.sig_xy = Function(self.U)

        #======= sigma_xy projection stuff:
        self.U_trial = TrialFunction(self.U)
        self.U_test = TestFunction(self.U)
        self.sig_form_xy = (2*self.mu*(sym(grad(self.u)) - self.eigenstrain))[0,1]
        self.a_sig_xy = inner(self.U_trial,self.U_test)*dx
        self.L_sig_xy = inner(self.sig_form_xy,self.U_test)*dx
        self.L_sig_xy_compiled = Form(inner(self.sig_form_xy,self.U_test)*dx)
        self.A_sig_xy = assemble(self.a_sig_xy)
        self.sig_xy_solver = KrylovSolver(self.A_sig_xy,'cg','hypre_euclid')
        self.sig_xy_solver.parameters['nonzero_initial_guess'] = True  
        self.sig_xy_solver.parameters['relative_tolerance'] = 1e-10
        self.sig_xy_solver.parameters['absolute_tolerance'] = 1e-12
        # self.b_sig_xy = assemble(self.L_sig_xy)
        self.b_sig_xy = PETScVector()
        #=======()
        #update the functions and make workspaces for sig_xy stuff:        
        self.V_trial, self.V_test = TrialFunction(self.V), TestFunction(self.V)
        #our u solver will splint the solve into two pieces
        eps = sym(grad(self.V_trial))
        fullForm = 2*self.mu*eps  + self.lmbda*tr(eps)*Identity(self.dim)
        plasticForm = 2*self.mu*self.eigenstrain  + self.lmbda*tr(self.eigenstrain)*Identity(self.dim)
        self.lhs_u = inner(fullForm, grad(self.V_test))*dx
        self.rhs_u = inner(plasticForm,grad(self.V_test))*dx 
        self.A_u = PETScMatrix()
        self.b_u = PETScVector()
        self.u_solver = KrylovSolver(self.A_u,'cg','hypre_euclid')
        self.u_solver.parameters['nonzero_initial_guess'] = True 
        self.u_solver.parameters['relative_tolerance'] = 1e-10
        self.u_solver.parameters['absolute_tolerance'] = 1e-12
        self.u_assembler = SystemAssembler(self.lhs_u,self.rhs_u,self.bc)

        #======== strain projection stuff:
        self.W_trial, self.W_test = TrialFunction(self.W), TestFunction(self.W)
        self.W_mass_form = inner(self.W_trial,self.W_test)*dx
        self.W_mass_form_compiled = Form(self.W_mass_form)
        self.eps_form = sym(grad(self.u))
        self.L_eps = inner(self.eps_form, self.W_test)*dx
        self.L_eps_compiled = Form(self.L_eps)
        self.A_eps = assemble(self.W_mass_form)
        self.b_eps = assemble(self.L_eps)
        self.W_solver = KrylovSolver(self.A_eps,'cg','hypre_euclid')
        self.W_solver.parameters['nonzero_initial_guess'] = True

        #stress projection stuff:
        self.sig_form = 2*self.mu*(sym(grad(self.u))-self.eigenstrain) + self.lmbda*tr(sym(grad(self.u)) - self.eigenstrain)*Identity(self.dim)
        self.L_sig = inner(self.sig_form, self.W_test)*dx 
        self.b_sig = assemble(self.L_sig)

        #inclusion projection form:
        self.stz_form = inner(self.W_trial,self.inclusionExpression2)*dx
        self.stz_form_compiled = Form(self.stz_form)


    def toDict(self):
        #save all stuff to a dict for pickling.dict = {}
        #except for eigenstrain, which we save to an HDF5 file
        dict = {}
        dict['N'] = self.gridN
        dict['wid'] = self.wid 
        dict['mu'] = self.mu.values()
        dict['lmbda'] = self.lmbda.values()
        dict['loadingCondition'] = self.loadingCondition 
        dict['loadingParam'] = self.loadingParam
        dict['plaq_thresh'] = self.plaq_thresh

        #building the eigenstrain array:
        eigstrain = np.zeros([self.cmap.num_plaq,4],order='f') #fortran order, so that columns are continguous
        for indx in range(0,4):
            self.Wmap.plaquette_average_mpi(self.eigenstrain.vector().get_local(),eigstrain[:,indx],indx)
        dict['eigenstrain'] = eigstrain
        if(self.cmap.isroot):
            return dict

    #broadcast the dict to all threads before use: 
    @classmethod
    def fromDict(cls,dict): 
        newepm = square(dict['N'], dict['loadingCondition'])
        newepm.mu.assign(dict['mu'])
        newepm.lmbda.assign(dict['lmbda'])
        newepm.plaq_thresh  = dict['plaq_thresh']
        newepm.loadingParam = dict['loadingParam']
        if(newepm.loadingCondition == 'fullPBC'):
            newepm.lastLoadingParam = newepm.loadingParam


        #NOTE: we are applying the same (constant) eigenstrain to all cells in each plaquette
        #NOTE: this works because we are using 2x2 cells / plaquette, all are identical.
        #NOTE: if we go to a finer mesh, we'll need to do more work. 
        for indx in range(0,4):
            newepm.Wmap.assign_plaquette_averages(newepm.eigenstrain.vector(),dict['eigenstrain'][:,indx], indx)
        
        return newepm 

    def setLoadingCondition(self, loadingCondition):
        self.loadingParam = 0.0
        def bottomBoundary(x,on_boundary):
            return (x[1] <= DOLFIN_EPS)
        def topBoundary(x,on_boundary):
            return (x[1] >= self.wid - DOLFIN_EPS)
        def topBotBoundary(x,on_boundary):
            return bottomBoundary(x,on_boundary) or topBoundary(x,on_boundary)
        def noBoundary(x,on_boundary):
            return False
        def allBoundary(x,on_boundary):
            return on_boundary
        self.loadingCondition = loadingCondition
      # Create mesh function over the cell facets
        self.boundary_subdomains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundary_subdomains.set_all(0)
        self.boundaryPressure = Constant((0.0,0.0))

        #each case MUST: define self.bc, and loadingFunc
        if(loadingCondition == 'tractionTop'):
            print('tractionTop not currently supported, needs -dot(boundaryPressure,self.V_test)*ds in RHS of plasticform')
            def loadingFunc(self):
                self.boundaryPressure.sigxy = self.loadingParam
            zeroVec = Constant((0.0,0.0))
            self.bc = DirichletBC(self.V,zeroVec,bottomBoundary)
            self.boundaryPressure = Expression(('sigxy','0.0'),sigxy=0.0,degree=0)
            AutoSubDomain(topBoundary).mark(self.boundary_subdomains, 1)
        elif(loadingCondition == 'simpleShearDisplacement'):
            self.simpleShearDisplacement = Expression(("x[1]*epsxy","0"), degree=1,epsxy = 0.00)
            self.bc = DirichletBC(self.V,self.simpleShearDisplacement,allBoundary)
            def loadingFunc(self):
                self.simpleShearDisplacement.epsxy = self.loadingParam
        elif(loadingCondition == 'fullPBC'):
            self.V = VectorFunctionSpace(self.mesh,'Lagrange',1,constrained_domain = FullPeriodicBoundary(self.wid))
            self.bc = []
            self.init_forms()
            self.lastLoadingParam = 0.
            def loadingFunc(self):
                deltaLoad = self.loadingParam - self.lastLoadingParam
                # print('loading: ' ,deltaLoad)
                #we use a negative below, because we want to apply negative eigenstrain? 
                eigv = self.eigenstrain.vector()
                eigv[np.arange(2,eigv.local_size(),4)] += -deltaLoad
                eigv[np.arange(1,eigv.local_size(),4)] += -deltaLoad
                self.lastLoadingParam = self.loadingParam
        elif(loadingCondition == 'simpleShearDisplacement_xPBC'):
            self.V = VectorFunctionSpace(self.mesh,'Lagrange',1,constrained_domain = PartialPeriodicBoundary(self.wid))
            self.simpleShearDisplacement = Expression(("x[1]*epsxy","0"), degree=1,epsxy = 0.00)
            self.bc = DirichletBC(self.V,self.simpleShearDisplacement,topBotBoundary)
            def loadingFunc(self):
                self.simpleShearDisplacement.epsxy = self.loadingParam
        elif(loadingCondition == 'simpleShearFree_x'):
            self.simpleShearDisplacement = Expression(("x[1]*epsxy","0"), degree=1,epsxy = 0.00)
            self.bc = DirichletBC(self.V,self.simpleShearDisplacement,topBotBoundary)
            def loadingFunc(self):
                self.simpleShearDisplacement.epsxy = self.loadingParam
                
        setattr(square,'applyLoading',loadingFunc)
        #setting the boundaries.
        self.dss = ds(subdomain_data=self.boundary_subdomains)

    def solve_u_new_bc(self):
        self.applyLoading()
        self.u_assembler.assemble(self.A_u,self.b_u)
        self.u_solver.set_operator(self.A_u)
        self.u_solver.solve(self.u.vector(),self.b_u)
 

    def solve_u_const_bc(self):
        self.u_assembler.assemble(self.b_u)
        self.u_solver.solve(self.u.vector(),self.b_u)
        

    def calceps(self):
        assemble(self.L_eps_compiled,self.b_eps)
        self.W_solver.solve(self.eps.vector(),self.b_eps)

    def calcsig(self):
        assemble(self.L_sig,self.b_sig)
        self.W_solver.solve(self.sig.vector(),self.b_sig)

    def calc_sig_xy(self):
        assemble(self.L_sig_xy_compiled,self.b_sig_xy)
        self.sig_xy_solver.solve(self.sig_xy.vector(),self.b_sig_xy)


    def addSTZ(self,plaq,mag=1.):
        dofstoapply = self.Wmap.plaq_dofs[plaq] # range(0,Wmap.plaq_dofs.lens[plaq])]
        cis = self.cmap.plaq_cells[plaq]
        self.inclusionExpression2.x0c, self.inclusionExpression2.x1c = self.cmap.plaq_centres[plaq]
        stz_delta = np.zeros(self.Wmap.dofs_per_cell*cis.shape[0])
        filled = 0
        for ci in cis:
            c = self.cmap.cs[ci]
            A = assemble_local(self.W_mass_form_compiled,c)
            b = assemble_local(self.stz_form_compiled,c)
            soln = np.linalg.solve(A,b)
            stz_delta[filled:filled+self.Wmap.dofs_per_cell] = soln
            filled+=self.Wmap.dofs_per_cell
        self.eigenstrain.vector()[dofstoapply] += stz_delta*mag


    def getSTZexpression(self):
        mag = self.inclusionExpression.strainmag
        nhatx = self.inclusionExpression.nhatx
        nhaty = self.inclusionExpression.nhaty        
        return np.array([
            mag*(2*nhatx*nhatx - 1),
            mag*2.0*nhatx*nhaty,
            mag*(2*nhaty*nhaty - 1)
        ])


    #here's the logic of our measurement codes:
    # get outputs a value, or, in the case of array valued guys (i.e. get_Y_plaqs)
    # outputs to the array. We output None in the case that we are not the root node
    # but, the logic should carry through with collective operations, even for non-root processes. 
    # however, if getting scalars, we'll return the result on all processes. 

    # also, all gets and all calcs should offer options to re-use cached values
    # order of calling should be: calc_sig_xy, calc_X(True), etc...




    #returns the x of the worst plaquette, and its index
    #will calc_X if not reusing plaq_X
    def get_worst_X(self,reuse_plaq_X = False,reuse_plaq_sig_xy=False):
        if(not reuse_plaq_X):
            self.calc_X_plaqs(reuse_plaq_sig_xy)
        pair = np.zeros(2)
        if(self.cmap.isroot):
            ind = np.argmin(self.plaq_x)
            pair[0] = self.plaq_x[ind]
            pair[1] = ind
        mpi4py.MPI.COMM_WORLD.Bcast(pair)
        return (pair[0],np.int32(pair[1])) 
        

    #compute sig_xy on a single plaq. Collective. 
    def get_plaq_sig_xy(self,ind,reuse_plaq_sig_xy=False):
        if(reuse_plaq_sig_xy):
            result = np.zeros(1)
            if(self.cmap.isroot):
                result[0] = self.plaq_sig_xy[ind]
            mpi4py.MPI.COMM_WORLD.Bcast(result)
            return result[0]
        else:
            return self.Umap.plaquette_by_ind(ind, self.sig_xy.vector().get_local())

    #computes X on a single plaquette. Collective.
    def get_plaq_X(self,ind,reuse_plaq_X=False):
        result = np.zeros(1)
        if(reuse_plaq_X):
            if(self.cmap.isroot):
                result[0] = self.plaq_x[ind] 
            mpi4py.MPI.COMM_WORLD.Bcast(result)
            return result[0]
        else: 
            return self.plaq_thresh[ind]-self.get_plaq_sig_xy(ind) 
            
            
    #Computes sig_xy on all plaquettes. collective. Does not calculate self.sig_xy
    def get_sig_xy_plaqs(self,array,reuse_plaq_sig_xy=False):
        if(reuse_plaq_sig_xy):
            if(self.cmap.isroot):
                np.copyto(array,self.plaq_sig_xy)
        else:
            self.Umap.plaquette_average_mpi(self.sig_xy.vector().get_local(),array=array)
        return array
    
    #sets array = thresh-abs(sig_xy) for all sites. Collective (if not reusing)
    def get_X_plaqs(self,array,reuse_plaq_X=False,reuse_plaq_sig_xy=False):
        if(reuse_plaq_X):
            if(self.cmap.isroot):
                np.copyto(array,self.plaq_x)
        else:
            self.get_sig_xy_plaqs(array,reuse_plaq_sig_xy)
            if(self.cmap.isroot):
                np.abs(array,out=array)
                np.negative(array,out=array)
                array+=self.plaq_thresh
        return array

    def calc_eps_xy_plaqs(self):
        self.Wmap.plaquette_average_mpi(self.eps.vector().get_local(),self.plaq_eps_xy,1)

    def calc_sig_xy_plaqs(self):
        self.get_sig_xy_plaqs(self.plaq_sig_xy)

    def calc_X_plaqs(self,reuse_plaq_sig_xy=False):
        self.get_X_plaqs(self.plaq_x,False,reuse_plaq_sig_xy)
    
    #calculates the x on each plaquette, without abs(sig_xy). Collective.
    def get_X_signed_plaqs(self,array,reuse_plaq_sig_xy=False):
        self.get_sig_xy_plaqs(array,reuse_plaq_sig_xy)
        if(self.cmap.isroot):
            np.negative(array,out=array)
            array+=self.plaq_thresh
        return array




    #===== old code below:
    def toDict_v01(self):
        #save all stuff to a dict for pickling.dict = {}  
        dict = {}
        dict['N'] = self.gridN
        dict['wid'] = self.wid 
        dict['mu'] = self.mu.values()
        dict['lmbda'] = self.lmbda.values()
        dict['loadingCondition'] = self.loadingCondition 
        dict['loadingParam'] = self.loadingParam
        dict['plaq_thresh'] = self.plaq_thresh
        dict['eigenstrain'] = self.eigenstrain.vector().gather_on_zero()
        if(self.cmap.isroot):
            return dict

    @classmethod
    def fromDict_v01(cls,dict): 
        newepm = square(dict['N'], dict['loadingCondition'])
        newepm.mu.assign(dict['mu'])
        newepm.lmbda.assign(dict['lmbda'])
        newepm.plaq_thresh  = dict['plaq_thresh']
        newepm.loadingParam = dict['loadingParam']
        eigV = newepm.eigenstrain.vector()
        rng = eigV.local_range()
        eigV[:] = dict['eigenstrain'][rng[0]:rng[1]]
        return newepm 

    def toDict_v02(self,fname_eigenstrain):
        #save all stuff to a dict for pickling.dict = {}
        #except for eigenstrain, which we save to an HDF5 file
        dict = {}
        dict['N'] = self.gridN
        dict['wid'] = self.wid 
        dict['mu'] = self.mu.values()
        dict['lmbda'] = self.lmbda.values()
        dict['loadingCondition'] = self.loadingCondition 
        dict['loadingParam'] = self.loadingParam
        dict['plaq_thresh'] = self.plaq_thresh

        hf = HDF5File(self.mesh.mpi_comm(),fname_eigenstrain,'w')
        hf.write(self.eigenstrain,'eigenstrain')
        printmpi()
        # self.mesh.mpi_comm().Barrier()
        # # hf.flush()
        # dict['eigenstrain'] = self.eigenstrain.vector().gather_on_zero()
        self.mesh.mpi_comm().Barrier()
        hf.close()
        if(self.cmap.isroot):
            return dict

    #broadcast the dict to all threads before use: 
    @classmethod
    def fromDict_v02(cls,dict,fname_eigenstrain): 
        newepm = square(dict['N'], dict['loadingCondition'])
        newepm.mu.assign(dict['mu'])
        newepm.lmbda.assign(dict['lmbda'])
        newepm.plaq_thresh  = dict['plaq_thresh']
        newepm.loadingParam = dict['loadingParam']
        if(newepm.loadingCondition == 'fullPBC'):
            newepm.lastLoadingParam = newepm.loadingParam
        printroot(fname_eigenstrain)
        hf = HDF5File(newepm.mesh.mpi_comm(),fname_eigenstrain,'r')
        hf.read(newepm.eigenstrain,'eigenstrain')
        hf.close()
        # eigV = newepm.eigenstrain.vector()
        # rng = eigV.local_range()
        # eigV[:] = dict['eigenstrain'][rng[0]:rng[1]]
        return newepm 



class FullPeriodicBoundary(SubDomain):
    def __init__(self,wid):
        super().__init__()
        self.L = wid 

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], self.L)) or 
                        (near(x[0], self.L) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        # if near(x[0], self.L) and near(x[1], self.L):
        #     y[0] = x[0] - self.L
        #     y[1] = x[1] - self.L
        # elif near(x[0], self.L):
        #     y[0] = x[0] - self.L
        #     y[1] = x[1]
        # else:   # near(x[1], self.L)
        #     y[0] = x[0]
        #     y[1] = x[1] - self.L
        if near(x[0],self.L):
            y[0] = x[0] - self.L
        else:
            y[0] = x[0]
        if near(x[1],self.L):
            y[1] = x[1] - self.L
        else:
            y[1] = x[1]


class PartialPeriodicBoundary(SubDomain):
    def __init__(self,wid):
        super().__init__()
        self.L = wid 

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left boundary
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        if near(x[0],self.L):
            y[0] = x[0] - self.L
        else:
            y[0] = x[0]
        y[1] = x[1]


    # def getMapSample(self,parameter,resolution=1):
    #     #resolution = 2 means you sample 4* as much data.
    #     #first, checking the dimension of the parameter, and building a corresponding grid:
    #     param_dim = np.size(parameter(self.wid/2,self.wid/2))
    #     numpoints = 1+resolution*self.gridN
    #     sample = np.zeros([numpoints,numpoints,param_dim])
    #     #now, sampling on the grid:
    #     for i in range(0,numpoints):
    #         for j in range(0,numpoints):
    #             x = i * self.wid/(numpoints-1)
    #             y = j * self.wid/(numpoints-1)
    #             sample[i,j] = parameter(x,y)
    #     return sample
        

    # def calcEquivSig(self):
    #     #requires an up to date self.sig.
    #     self.sigdev.assign( dklib.projfunctions.mixedproject( (self.sig - tr(self.sig)/self.dim*Identity(self.dim)),self.W))
    #     #self.equivStress.assign( project(sqrt(3/2* self.sigdev **2) ,self.U) )
    #     #do we want the 3/2? Not sure where it's from, so let's consider omitting it for now.
    #     #is 3 from dimension? Then we want 2/2 = 1. 
    #     self.equivStress.assign( project(sqrt(self.sigdev **2) ,self.U))

    # def cache_stz(self,plaq_ind = 0):
    #     #builds the STZ applied to (by default) plaquette zero
    #     #caches the dof change, for application to other sites. 
    #     self.inclusionExpression2.x0c, self.inclusionExpression2.x1c = self.cmap.plaq_centres[plaq_ind]
    #     self.cached_stz_delta = np.zeros(np.shape(self.Wmap.plaq_dofs[plaq_ind,:]))
    #     filled = 0
    #     for cind in self.cmap.plaq_cells[plaq_ind,:]:
    #         c = self.cmap.cs[cind]
    #         A = assemble_local(self.W_mass_form,c)
    #         b = assemble_local(self.stz_form,c)
    #         soln = np.linalg.solve(A,b)
    #         ndof = np.size(soln)
    #         self.cached_stz_delta[filled:filled+ndof] = soln 
    #         filled+=ndof  

    # def addSTZ(self,plaq_ind,mag=1.):
    #     self.eigenstrain.vector()[self.Wmap.plaq_dofs[plaq_ind,:]] += mag*self.cached_stz_delta

    # def addSTZ(self,x,y,rad,mag):
    #     self.inclusionExpression.radsq = rad*rad
    #     self.inclusionExpression.strainmag = mag / np.sqrt(2.) #over sqrt 2, so that deltaSigma_ij deltaSigma_ij = mag. 
    #     self.inclusionExpression.x0c = x
    #     self.inclusionExpression.x1c = y
    #     self.eigenstrain.assign(dklib.projfunctions.mixedproject(self.eigenstrain+self.inclusionExpression,self.W))

    # def addAssocSTZ(self,x,y,rad,mag):
    #     self.inclusionExpression2.x0c = x
    #     self.inclusionExpression2.x1c = y
    #     self.inclusionExpression2.radsq = rad*rad
    #     dev = self.sigdev(x,y)
    #     self.inclusionExpression2.sigxx = dev[0] #/ self.equivStress(x,y) #commented out, b/c moved up into strainmag
    #     self.inclusionExpression2.sigxy = dev[1] #/ self.equivStress(x,y)
    #     self.inclusionExpression2.sigyy = dev[3] #/ self.equivStress(x,y)
    #     self.inclusionExpression2.strainmag = mag / self.equivStress(x,y)
    #     #self.inclusionExpression2.strainmag = mag / np.sqrt(.5*(dev[0]**2 + 2*dev[1]**2 + dev[3]**2))#self.equivStress(x,y)
    #     self.eigenstrain.assign(dklib.projfunctions.mixedproject(self.eigenstrain+self.inclusionExpression2,self.W))
    #     #return np.array([self.inclusionExpression2.sigxx,  self.inclusionExpression2.sigxy,  self.inclusionExpression2.sigyy])


    # def getEquivX(self):
    #     N = np.shape(self.siteCoords)[0]
    #     data = np.zeros(N)
    #     for i in range(0,N):
    #         data[i] = self.equivStress(self.siteCoords[i]) - self.thresholds[i]
    #     return data

    # def getXabs(self):
    #     N = np.shape(self.siteCoords)[0]
    #     data = np.zeros(N)
    #     for i in range(0,N):
    #         data[i] = np.abs(self.sig(self.siteCoords[i])[1]) - self.thresholds[i]
    #     return data
        
    
    # def getEquivSites(self):
    #     N = np.shape(self.siteCoords)[0]
    #     data = np.zeros(N)
    #     for i in range(0,N):
    #         data[i] = self.equivStress(self.siteCoords[i])
    #     return data 
    # def getAssocSTZexpression(self):
    #     mag = self.inclusionExpression2.strainmag
    #     return mag*np.array([self.inclusionExpression2.sigxx,  self.inclusionExpression2.sigxy,  self.inclusionExpression2.sigyy])

    # def getMaxEquivSigOver(self):
    #     maxSigIndex = np.argmax([self.equivStress(coord)-self.thresholds[ind] for ind, coord in enumerate(self.siteCoords)])
    #     return self.equivStress(self.siteCoords[maxSigIndex])-self.thresholds[maxSigIndex],maxSigIndex

    # def getMaxSigOver(self):
    #     maxSigIndex = np.argmax([self.sig(coord)[1]-self.thresholds[ind] for ind, coord in enumerate(self.siteCoords)])
    #     return self.sig(self.siteCoords[maxSigIndex])[1]-self.thresholds[maxSigIndex],maxSigIndex
