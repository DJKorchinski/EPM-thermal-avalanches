import numpy as np 

class hist2d:
    def __init__(self,bins1,bins2):
        self.bins1 = bins1 
        self.bins2 = bins2
        self.Nbins1 = np.shape(bins1)[0]-1
        self.Nbins2 = np.shape(bins2)[0]-1
        self.counts = np.zeros((self.Nbins1,self.Nbins2),dtype=np.int64)
    
    def addData(self,dat1,dat2):
        counts,edges1,edges2 = np.histogram2d(dat1,dat2,[self.bins1,self.bins2])
        self.counts+=np.int64(counts)

def normHist(hist):
    pdf = np.array(hist.counts,dtype = np.float64)
    pdf /= np.sum(hist.counts)
    for i in range(0,np.shape(pdf)[0]):
        pdf[i,:] /= hist.bins2[1:hist.bins2.size] - hist.bins2[:hist.bins2.size-1]
    for i in range(0,np.shape(pdf)[1]):
        pdf[:,i] /= hist.bins1[1:hist.bins1.size] - hist.bins1[:hist.bins1.size-1]
    return pdf