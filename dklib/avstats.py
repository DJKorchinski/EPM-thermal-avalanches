import dklib.avalanche
import numpy as np;
#we don't always have pyplot. If we lack it, just ignore it -- we hopefully won't call any methods that require pyplot.
try:
    import matplotlib.pyplot as plt;
except ImportError as e:
    pass

def addCountsToHist(binedges,counts,data):
    #adds data to the counts as a  histogram
    values,binedges = np.histogram(data,bins = binedges)
    for i in range(0,counts.size):
        counts[i] = counts[i] + values[i]

def normHist(binedges,counts):
    density = counts.copy()
    density /= ((binedges[1:binedges.size]-binedges[:binedges.size-1]) * np.sum(counts))
    return density


def sizeDistribution(avalanches,existingBins=None,existingValues=None,plotDensity=False,legend=None):
    #=============
    #returns bins, counts, density for histogram.
    #=============
    if(existingBins is None):
        bins = buildBins(1,10**5,50)
    else:
        bins = existingBins

    if(existingValues is None):
        counts = np.zeros(bins.size-1)
    else:
        counts = existingValues
    
    sizes = [a.numSites for a in avalanches]
    addCountsToHist(bins,counts,sizes)
    density = normHist(bins,counts)

    if(plotDensity):
        plotSizeDistribution(bins,density,legends=legend)
    
    return bins,counts,density



def plotSizeDistribution(bins,density,title='Failing sites in avalanches',legends=None,plotpl = False):
    bincenters = binCenters(bins)
    shp = np.shape(density)
    if(np.size(shp) > 1):
        for i in range(0,shp[0]):
            if(legends is None):
                plt.plot(bincenters,density[i])
            else:
                plt.plot(bincenters,density[i],label = legends[i])
    else:
        if(legends is None):
            plt.plot(bincenters,density)
        else:
            plt.plot(bincenters,density,label = legends)
    if(plotpl):
        plt.plot(bincenters,np.power(bincenters,-4./3),label = 'x^(-4/3)')
        plt.plot(bincenters,np.power(bincenters,-1.8),label = 'x^(-1.8)')
    plt.title(title)
    plt.legend()
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.ylabel('P(S) (density)')
    plt.xlabel('Size (number of failing sites)')
#    plt.show()
    

def stressDropDistribution(avalanches,existingBins=None,existingValues=None,plotDensity=False):
    #=============
    #returns bins, counts, density for histogram.
    #=============
    if(existingBins==None):
        bins = buildBins(10**-5,10**0,50,False)
    else:
        bins = existingBins

    if(existingValues==None):
        counts = np.zeros(bins.size-1)
    else:
        counts = existingValues

    sizes = [-a.vsstress2+a.vsstress1 for a in avalanches]#sign flip so that the stress drop is postivie.
    addCountsToHist(bins,counts,sizes)
    density = normHist(bins,counts)

    if(plotDensity):
        plotStressDistribution(bins,density)
    
    return bins,counts,density

def boundaryStressDropDistribution(avalanches,existingBins=None,existingValues=None,plotDensity=False):
    #=============
    #returns bins, counts, density for histogram.
    #=============
    if(existingBins is None):
        bins = buildBins(10**-5,10**0,50,False)
    else:
        bins = existingBins

    if(existingValues is None):
        counts = np.zeros(bins.size-1)
    else:
        counts = existingValues

    sizes = [-a.bsstress2+a.bsstress1 for a in avalanches]
    addCountsToHist(bins,counts,sizes)
    density = normHist(bins,counts)

    if(plotDensity):
        plotStressDistribution(bins,density,'Boundary stress drop during avalanche')
    
    return bins,counts,density

def plotStressDistribution(bins,density, title='Volume stress drop during avalanche'):
    bincenters = binCenters(bins,False)
    plt.plot(bincenters,density,bincenters,np.power(bincenters,-1.333),bincenters,np.power(bincenters,-1.8))
    plt.legend(['data','x^{-1.333}','x^{-1.8}'])
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.ylabel('P(stress drop) (density)')
    plt.xlabel('Stress drop')
    plt.show()


def pairedStressStrains(avalanches):
    stresses2 = [a.vsstress2 for a in avalanches]
    stresses1 = [a.vsstress1 for a in avalanches]
    pairedVSStresses = np.zeros(2*len(stresses2))
    pairedVSStresses[::2] = stresses1
    pairedVSStresses[1::2] = stresses2
    stresses2 = [a.bsstress2 for a in avalanches]
    stresses1 = [a.bsstress1 for a in avalanches]
    pairedBSStresses = np.zeros(2*len(stresses2))
    pairedBSStresses[::2] = stresses1
    pairedBSStresses[1::2] = stresses2
    stresses2 = [a.vsstrain2 for a in avalanches]
    stresses1 = [a.vsstrain1 for a in avalanches]
    pairedVSStrains = np.zeros(2*len(stresses2))
    pairedVSStrains[::2] = stresses1
    pairedVSStrains[1::2] = stresses2
    return pairedVSStresses,pairedVSStrains,pairedBSStresses

def plotStressStrains(pairedVSStresses,pairedVSStrains,pairedBSStresses):
    plt.plot(pairedVSStrains,pairedVSStresses)
    plt.xlabel('Volume strain')
    plt.ylabel('Volume stress')
    plt.show()
    plt.plot(pairedVSStrains,pairedBSStresses)
    plt.xlabel('Volume strain')
    plt.ylabel('Boundary stress')
    plt.show()
    plt.plot(pairedVSStresses,pairedBSStresses)
    plt.xlabel('Volume stress')
    plt.ylabel('Boundary stress')
    plt.show()



#only accepts a single row of data and a single label.
def buildLogLog(bins,data,labels = None, plotAfter=False, intBinning=True):   
    bincenters = binCenters(bins,intBinning)
    if(labels is None):
        plt.plot(bincenters,data)
    else:
        plt.plot(bincenters,data,label = labels)
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    if(plotAfter):
        plot.show()
