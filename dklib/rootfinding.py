from numpy import sign,sqrt;

def riddersMethod(func,x0,x1,x0v=None,x1v=None,xpres = 1e-12,ypres = 1e-12,maxiter = 30,debglabel = -1,args = []):
    iters = 0
    if(x0v is None):
        x0v = func(x0,*args)
    if(x1v is None):
        x1v = func(x1,*args)

    if(x0v*x1v > 0):
        print("root finding not initially bracketed.",x0,x0v,x1,x1v)
        return (x0+x1)/2

    while (abs(x0-x1) > xpres and abs(x0v-x1v) > ypres and iters < maxiter):        
        iters+=1
        x2 = (x0+x1)/2
        x2v = func(x2,*args)
        x3 = x2 + (x2-x0)*sign(x0v)*x2v / sqrt(x2v*x2v - x0v*x1v)
        x3v = func(x3,*args)

        #print(x0,x1,x2,x3,x0v,x1v,x2v,x3v)
        #print(abs(x3v), ypres, abs(x3v)<ypres)
        
        if(abs(x3v) < ypres):
            #print('rootfinding took: ',iters)
            return x3

        if(x3v*x2v < 0):
            x0v = x2v
            x0 = x2
            x1v = x3v
            x1 = x3
        elif(x3v*x0v<0):
            x1v = x3v
            x1 = x3
        elif(x3v*x1v<0):
            x0 = x3
            x0v = x3v
        else:
            print('weird failure?')
            print(iters,x0,x0v,x1,x1v,x2,x2v,x3,x3v)
            if(x2v == 0):
                return x2 
            elif(x3v == 0):
                return x3
        #print(debglabel,iters,' ',x0,x1,abs(x0-x1),(x0v),(x1v),abs(x0v-x1v))
    
    #print('rootfinding took: ',iters)
    return (x0+x1)/2
