from dklib.mpi_utils import printroot

class progressBar:
    def __init__(self,interval):
        self.lastPrint,self.printInterval = -1,interval
    def update(self,iter,maxiter,printfunc = printroot,prepend = ''):
        wouldPrint = (int( (100*iter) / (maxiter-1)) // self.printInterval ) * self.printInterval
        if(wouldPrint != self.lastPrint):
            printfunc(prepend+'%d '%wouldPrint+'%')
            self.lastPrint = wouldPrint
    