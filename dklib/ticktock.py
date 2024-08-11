import time 
def tick():
    global t1
    t1 = time.perf_counter_ns()
def tock(prefix='',postfix='',printout=True,printfunc=print):
    global t1
    t2 = time.perf_counter_ns()
    dt = (t2-t1)*1e-9
    if(printout):
        printfunc(prefix,dt,'seconds',postfix)
    return dt

class timer:
    def __init__(self):
        self.t1 = time.perf_counter_ns()
        self.total = 0.0
        self.N_tocks = 0
    def tick(self):
        self.t1 = time.perf_counter_ns()
    def tock(self,printout=False,prefix='',postfix='',printfunc=print):
        dt =  self.seconds_since_tick() 
        self.total+=dt
        self.N_tocks+=1
        if(printout):
            printfunc(prefix,self.total,'+',dt,'seconds',postfix)
    def seconds_since_tick(self):
        return (time.perf_counter_ns() -self.t1)*1e-9
    def chime(self,prefix='',postfix='',printfunc=print):
        printfunc(prefix,self.total,'seconds in ',self.N_tocks,'periods averaging',self.total/self.N_tocks,postfix)