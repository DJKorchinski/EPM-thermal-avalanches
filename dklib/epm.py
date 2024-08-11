from dolfin import * # pylint: disable=unused-wildcard-import
import dklib.projfunctions
import numpy as np
#from mpi4py import MPI 


#Contains a data structure to describe and methods to evolve a single square of elasto-plastic material.
#Has methods to sample parameters from the EPM, to solve the elastic equations associated with the problem, and maintain thresholds for failure.
class square:
    def __init__(self,gridN,wid,loadingCondition=None):
        self.gridN = gridN
        self.wid = wid
        self.mesh = RectangleMesh(MPI.comm_self,Point(0.,0.), Point(wid,wid),gridN,gridN,"crossed")
#        self.mesh = RectangleMesh(Point(0.,0.), Point(wid,wid),gridN,gridN,"crossed")
        self.dim = self.mesh.geometry().dim()
        self.mu, self.lmbda = Constant(1.), Constant(1.)
        self.U = FunctionSpace(self.mesh,'DG',0)
        self.W = TensorFunctionSpace(self.mesh,'DG',0,shape=(self.dim,self.dim))#,symmetry=True) 
        self.eps = Function(self.W,name="Strain") 
        self.sig = Function(self.W,name="Stress")
        self.sigdev = Function(self.W,name="Deviatoric Stress")
        self.eigenstrain = interpolate(Constant(((0,0),(0,0))),self.W)
        self.equivStress = interpolate(Constant(0),self.U)
        self.V = VectorFunctionSpace(self.mesh,'Lagrange',1)
        self.thresholds = np.ones(self.mesh.num_vertices()) # 1.0-self.r.random_sample(self.mesh.num_vertices())
        self.siteCoords = self.mesh.coordinates()


        if(loadingCondition != None):
            self.setLoadingCondition(loadingCondition)            
        
        #build the forms for assembly:
        self.init_forms()
        

        #initializing the inclusion expression:
        phi = np.pi / 4.0
        self.inclusionExpression = Expression((("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*(2*nhatx*nhatx - 1):0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*2.0*nhatx*nhaty:0.0"),("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*2.0*nhatx*nhaty:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*(2*nhaty*nhaty - 1):0.0")),degree=2,x0c=0,x1c=0,radsq=100,nhatx=cos(phi),nhaty=sin(phi),strainmag = 0.0)
        #set a raw (sigxx, sigxy, sigyy) type matrix expression:
        self.inclusionExpression2 = Expression((("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxx:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxy:0.0"),("(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigxy:0.0","(((x[0]-x0c)*(x[0]-x0c)+(x[1]-x1c)*(x[1]-x1c))<radsq)?strainmag*sigyy:0.0")),degree=6,x0c=0,x1c=0,radsq=100,sigxx = 0.,sigxy =0.,sigyy=0.,strainmag = 0.0)

    def init_forms(self):
        self.u = Function(self.V)
        self.sig_xy = Function(self.U)

        #======= sigma_xy projection stuff:
        self.U_trial = TrialFunction(self.U)
        self.U_test = TestFunction(self.U)
        self.sig_form_xy = (2*self.mu*(sym(grad(self.u)) - self.eigenstrain))[0,1]
        self.a_sig_xy = inner(self.U_trial,self.U_test)*dx
        self.L_sig_xy = inner(self.sig_form_xy,self.U_test)*dx
        self.sig_xy_solver = KrylovSolver('cg','ilu')
        self.sig_xy_solver.parameters['nonzero_initial_guess'] = True  
        self.sig_xy_solver.parameters['relative_tolerance'] = 1e-9 
        self.sig_xy_solver.parameters['absolute_tolerance'] = 1e-12
        self.A_sig_xy = assemble(self.a_sig_xy)
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
        self.u_solver = KrylovSolver('cg','ilu')
        self.u_solver.parameters['nonzero_initial_guess'] = True 
        self.u_solver.parameters['relative_tolerance'] = 1e-9 
        self.u_solver.parameters['absolute_tolerance'] = 1e-12 
        self.u_solver.parameters['nonzero_initial_guess'] = True 
        self.u_assembler = SystemAssembler(self.lhs_u,self.rhs_u,self.bc)

        

    def toDict(self):
        #save all stuff to a dict for pickling.dict = {}  
        dict['gridN'] = self.gridN 
        dict['wid'] = self.wid 
        dict['mu'] = self.mu.values()
        dict['lmbda'] = self.lmbda.values()
        dict['loadingCondition'] = self.loadingCondition 
        dict['thresholds'] = self.thresholds 
        dict['siteCoords'] = self.siteCoords
        dict['loadingParam'] = self.loadingParam
        dict['eigenstrain'] = np.array(self.eigenstrain.vector())
        return dict

    @classmethod
    def fromDict(cls,dict): 
        #call epm = epm.square.fromDict(dict)
        #reconstitutes an epm from a dict creatd by 'add to dict'
        newepm = square(dict['gridN'], dict['wid'], dict['loadingCondition'])
        newepm.mu.assign(dict['mu'])
        newepm.lmbda.assign(dict['lmbda'])
        newepm.thresholds  = dict['thresholds']
        newepm.siteCoords = dict['siteCoords']
        newepm.loadingParam = dict['loadingParam']
        newepm.eigenstrain.vector()[:] = dict['eigenstrain']
        return newepm 

    def setLoadingCondition(self, loadingCondition):
        self.loadingParam = 0.0
        def bottomBoundary(x,on_boundary):
            return (x[1] <= DOLFIN_EPS)
        def topBoundary(x,on_boundary):
            return (x[1] >= self.wid - DOLFIN_EPS)
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

        setattr(square,'applyLoading',loadingFunc)
        #setting the boundaries.
        self.dss = ds(subdomain_data=self.boundary_subdomains)

    def solve_u_new_bc(self):
        self.applyLoading()
        self.u_assembler.assemble(self.A_u,self.b_u)
        self.u_solver.solve(self.A_u,self.u.vector(),self.b_u)
 

    def solve_u_const_bc(self):
        self.u_assembler.assemble(self.b_u)
        self.u_solver.solve(self.A_u,self.u.vector(),self.b_u)
        # solve(self.A_u,self.u.vector(),self.b_u_pl+self.b_u_bc)
        

    def calceps(self):
        self.eps.assign(dklib.projfunctions.mixedproject(sym(grad(self.u)),self.W))

    def calcsig(self):
        self.sig.assign(dklib.projfunctions.mixedproject(2*self.mu*(sym(grad(self.u))-self.eigenstrain) + self.lmbda*tr(sym(grad(self.u)) - self.eigenstrain)*Identity(self.dim),self.W))

    def calc_sig_xy(self):
        assemble(self.L_sig_xy,self.b_sig_xy)
        self.sig_xy_solver.solve(self.A_sig_xy,self.sig_xy.vector(),self.b_sig_xy)
        # solve(self.A_sig_xy,self.sig_xy.vector(),self.b_sig_xy)
        

    def calcEquivSig(self):
        #requires an up to date self.sig.
        self.sigdev.assign( dklib.projfunctions.mixedproject( (self.sig - tr(self.sig)/self.dim*Identity(self.dim)),self.W))
        #self.equivStress.assign( project(sqrt(3/2* self.sigdev **2) ,self.U) )
        #do we want the 3/2? Not sure where it's from, so let's consider omitting it for now.
        #is 3 from dimension? Then we want 2/2 = 1. 
        self.equivStress.assign( project(sqrt(self.sigdev **2) ,self.U))

    def addSTZ(self,x,y,rad,mag):
        self.inclusionExpression.radsq = rad*rad
        self.inclusionExpression.strainmag = mag / np.sqrt(2.) #over sqrt 2, so that deltaSigma_ij deltaSigma_ij = mag. 
        self.inclusionExpression.x0c = x
        self.inclusionExpression.x1c = y
        self.eigenstrain.assign(dklib.projfunctions.mixedproject(self.eigenstrain+self.inclusionExpression,self.W))

    def addAssocSTZ(self,x,y,rad,mag):
        self.inclusionExpression2.x0c = x
        self.inclusionExpression2.x1c = y
        self.inclusionExpression2.radsq = rad*rad
        dev = self.sigdev(x,y)
        self.inclusionExpression2.sigxx = dev[0] #/ self.equivStress(x,y) #commented out, b/c moved up into strainmag
        self.inclusionExpression2.sigxy = dev[1] #/ self.equivStress(x,y)
        self.inclusionExpression2.sigyy = dev[3] #/ self.equivStress(x,y)
        self.inclusionExpression2.strainmag = mag / self.equivStress(x,y)
        #self.inclusionExpression2.strainmag = mag / np.sqrt(.5*(dev[0]**2 + 2*dev[1]**2 + dev[3]**2))#self.equivStress(x,y)
        self.eigenstrain.assign(dklib.projfunctions.mixedproject(self.eigenstrain+self.inclusionExpression2,self.W))
        #return np.array([self.inclusionExpression2.sigxx,  self.inclusionExpression2.sigxy,  self.inclusionExpression2.sigyy])

    def getSTZexpression(self):
        mag = self.inclusionExpression.strainmag
        nhatx = self.inclusionExpression.nhatx
        nhaty = self.inclusionExpression.nhaty        
        return np.array([
            mag*(2*nhatx*nhatx - 1),
            mag*2.0*nhatx*nhaty,
            mag*(2*nhaty*nhaty - 1)
        ])

    def getAssocSTZexpression(self):
        mag = self.inclusionExpression2.strainmag
        return mag*np.array([self.inclusionExpression2.sigxx,  self.inclusionExpression2.sigxy,  self.inclusionExpression2.sigyy])


    def setSites(self,newSiteCoordinates,newThresholds = None):
        self.siteCoords = newSiteCoordinates
        self.siteN = np.shape(newSiteCoordinates)[0]
        self.x = np.ones(self.siteN)
        if(newThresholds is None):
            self.thresholds = np.ones(self.siteN)
        else:
            self.thresholds = newThresholds
        

    def getMaxEquivSigOverVertices(self):
        maxSigIndex = np.argmax([self.equivStress(coord)-self.thresholds[ind] for ind, coord in enumerate(self.mesh.coordinates())])
        return self.equivStress(self.mesh.coordinates()[maxSigIndex])-self.thresholds[maxSigIndex],maxSigIndex

    def getMaxEquivSigOver(self):
        maxSigIndex = np.argmax([self.equivStress(coord)-self.thresholds[ind] for ind, coord in enumerate(self.siteCoords)])
        return self.equivStress(self.siteCoords[maxSigIndex])-self.thresholds[maxSigIndex],maxSigIndex


    #returns max(sigma-sigma_y), and the index of vertex at which this occurs.
    def getMaxSigOverVertices(self):
        maxSigIndex = np.argmax([self.sig(coord)[1]-self.thresholds[ind] for ind, coord in enumerate(self.mesh.coordinates())])
        return self.sig(self.mesh.coordinates()[maxSigIndex])[1]-self.thresholds[maxSigIndex],maxSigIndex

    def getMaxSigOver(self):
        maxSigIndex = np.argmax([self.sig(coord)[1]-self.thresholds[ind] for ind, coord in enumerate(self.siteCoords)])
        return self.sig(self.siteCoords[maxSigIndex])[1]-self.thresholds[maxSigIndex],maxSigIndex

    def getMaxAbsSigOver(self):
        maxSigIndex = np.argmax(self.x)
        return self.x[maxSigIndex],maxSigIndex

    #returns sig-sig_y for all sites
    def calcX(self):
        for i in range(0,self.siteN):
            self.x[i] = np.abs(self.sig_xy(self.siteCoords[i])) - self.thresholds[i]

    def getEquivX(self):
        N = np.shape(self.siteCoords)[0]
        data = np.zeros(N)
        for i in range(0,N):
            data[i] = self.equivStress(self.siteCoords[i]) - self.thresholds[i]
        return data

    def getXabs(self):
        N = np.shape(self.siteCoords)[0]
        data = np.zeros(N)
        for i in range(0,N):
            data[i] = np.abs(self.sig(self.siteCoords[i])[1]) - self.thresholds[i]
        return data
        
    
    def getEquivSites(self):
        N = np.shape(self.siteCoords)[0]
        data = np.zeros(N)
        for i in range(0,N):
            data[i] = self.equivStress(self.siteCoords[i])
        return data 

    def getSigSites(self):
        N = np.shape(self.siteCoords)[0]
        data = np.zeros(N)
        for i in range(0,N):
            data[i] = self.sig(self.siteCoords[i])[1]
        return data 


    def getMapSample(self,parameter,resolution=1):
        #resolution = 2 means you sample 4* as much data.
        #first, checking the dimension of the parameter, and building a corresponding grid:
        param_dim = np.size(parameter(self.wid/2,self.wid/2))
        numpoints = 1+resolution*self.gridN
        sample = np.zeros([numpoints,numpoints,param_dim])
        #now, sampling on the grid:
        for i in range(0,numpoints):
            for j in range(0,numpoints):
                x = i * self.wid/(numpoints-1)
                y = j * self.wid/(numpoints-1)
                sample[i,j] = parameter(x,y)
        return sample
        
