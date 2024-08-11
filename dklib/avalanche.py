from dolfin import assemble,dx,ds
import numpy as np
#Contains a data structure for holding statistics about a single avalanche.

class Avalanche:
    def __init__(self,numsites=1):
        self.numSites=numsites
        self.vsstrain1 = 0 #volume shear strain, before avalanche.
        self.vsstrain2 = 0 #volume shear strain, after avalanche.
        self.vsstress1 = 0
        self.vsstress2 = 0
        self.bsstress1 = 0 #boundary shear stress, before
        self.bsstress2 = 0 #boundary shear stress, after
        self.vestrain1 = 0 #volume eigenstrain, before
        self.vestrain2 = 0
        self.equivstress1 = 0
        self.equivstress2 = 0
        self.loadparam = 0 #generic loading parameter, that we use to drive our simulation?

    def setMeasureablesIntegrate(self,mesh,strain,stress,eigstrain,numset,equivStress=None):
        area = assemble(1.0*dx(domain=mesh))
        perimeter = assemble(1.0*ds(domain=mesh))
        vsstrain = assemble(strain[0,1]*dx(domain=mesh)) / area
        vsstress = assemble(stress[0,1]*dx(domain=mesh)) / area
        bsstress = assemble(stress[0,1]*ds(domain=mesh)) / perimeter 
        vestrain = assemble(eigstrain[0,1]*dx(domain=mesh)) / area
        if(equivStress!=None):
            equivstress = assemble(equivStress*dx(domain=mesh)) / area

        if(numset == 1):
            self.vsstrain1 = vsstrain
            self.vsstress1 = vsstress
            self.bsstress1 = bsstress
            self.vestrain1 = vestrain
            self.equivstress1 = equivstress
        elif(numset == 2):
            self.vsstrain2 = vsstrain
            self.vsstress2 = vsstress
            self.bsstress2 = bsstress
            self.vestrain2 = vestrain
            self.equivstress2 = equivstress
        else:
            print('incorect numset chosen in "setMeasureables" function.')


    def setMeasureables(self,siteCoords,strain,stress,eigstrain,numset,equivStress=None):
        vsstrain = np.mean(np.array([strain(coord) for coord in siteCoords]))
        vsstress = np.mean(np.array([stress(coord) for coord in siteCoords]))
        vestrain = np.mean(np.array([eigstrain(coord) for coord in siteCoords]))

        if(not equivStress is None):
            vequiv = np.mean(np.array([equivStress(coord) for coord in siteCoords]))

        if(numset == 1):
            self.vsstrain1 = vsstrain
            self.vsstress1 = vsstress
            self.vestrain1 = vestrain
            if(not equivStress is None):
                self.equivstress1 = vequiv
        elif(numset == 2):
            self.vsstrain2 = vsstrain
            self.vsstress2 = vsstress
            self.vestrain2 = vestrain
            if(not equivStress is None):
                self.equivstress2 = vequiv
        else:
            print('incorect numset chosen in "setMeasureables" function.')



