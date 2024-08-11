try:
    import matplotlib.pyplot as plt;
except ImportError as e:
    print('using dklib.plothelp without matplotlib')
    pass
import numpy as np


def logx(ax=None):
    if(ax is None):
        ax = plt.gca()
    ax.set_xscale('log')
def logy(ax = None ):
    if(ax is None):
        ax = plt.gca()
    ax.set_yscale('log')
def loglog(ax = None ):
    logx(ax)
    logy(ax)
def linx(ax=None):
    if(ax is None):
        ax = plt.gca()
    ax.set_xscale('linear')
def liny(ax=None):
    if(ax is None):
        ax = plt.gca()
    ax.set_yscale('linear')
def linlin(ax = None):
    linx(ax)
    liny(ax)

def plCut(coef,exponent,min,max,src = plt,**kwargs):
    pts = np.logspace(np.log10(min),np.log10(max))
    src.plot(pts,coef*pts**exponent, **kwargs)

def labelledPlCut(text,ymultoffset,xmultoffset=1.0,*args,**kwargs):
    src = plt
    if('src' in kwargs):
        src = kwargs['src']
    textkwargs={}
    linekwargs = dict(kwargs)
    if('textkwargs' in kwargs):
        textkwargs = kwargs['textkwargs']
        del linekwargs['textkwargs']
    plCut(*args,**linekwargs)
    src.text(xmultoffset*(args[2]*args[3])**.5,args[0] * (args[2]*args[3])**(args[1]*.5) * ymultoffset,text,**textkwargs)

def plFit(xs,ys,xsamples = None,src = plt,labelExponent = False,**kwargs):
    if(xsamples is None):
        xsamples = xs 
    lgfit = lambda x,poly: np.exp(poly[1])*x**poly[0]
    xlog = np.log(xs)
    ylog = np.log(ys)
    poly = np.polyfit(xlog,ylog,1)
    if('label' not in kwargs.keys()):
        kwargs['label'] = '$x^{%.3g}$'%poly[0]
    else: 
        if(labelExponent):
            kwargs['label'] = kwargs['label']%poly[0]
    return poly,src.plot(xsamples, lgfit(xsamples,poly),**kwargs)
    

#adds a point to an existing line. 
def addPt(xs,pxs,xpt,src = plt,**kwargs):
    ypt = np.interp(xpt,xs,pxs)
    src.plot(xpt,ypt,**kwargs)
    return xpt,ypt
    
def buildGeomBins(lowerBound,upperBound,numBins=50,integerBoundaries=False):
    #builds bins nicely
    if(integerBoundaries):
        bins = np.unique(np.round(np.geomspace(lowerBound,upperBound,numBins)))
#            np.logspace(np.log10(lowerBound),np.log10(upperBound),numBins)))
    else:
        bins = np.geomspace(lowerBound,upperBound,numBins)
    return bins
        
def binCenters(binedges,excludeUpperInteger=False,geometric=True):
    #gives the middle of bins, with the integer accomodation if necessary
    if(geometric):
        if(excludeUpperInteger):
            return np.sqrt(np.abs(binedges[:binedges.size-1]*(binedges[1:binedges.size]-1)))
        else:
            return np.sqrt(np.abs(binedges[:binedges.size-1]*(binedges[1:binedges.size])))
    else:
        return 0.5*(binedges[:binedges.size-1] + binedges[1:binedges.size]) 

def getNearestBin(binedges,target):
    return np.clip(np.searchsorted(binedges,target)-1,0,binedges.size-1)

def normHist(binedges,counts):
    density = np.array(counts,dtype=np.float64,copy=True)
    density /= ((binedges[1:binedges.size]-binedges[:binedges.size-1]) * np.sum(counts))
    return density

def probDensity(binedges,events,geom=True,excludeUpperInteger=False):
    counts,_ = np.histogram(events,binedges)
    return binCenters(binedges,excludeUpperInteger,geom),normHist(binedges,counts)

def histmean(binedges,density=None,counts=None,excludeUpperInteger=False,geometric=True,binC=None):
    if(density is None):
        density = normHist(binedges,counts)
    if(binC is None):
        binC = binCenters(binedges,excludeUpperInteger,geometric)
    return np.nansum((binedges[1:]-binedges[:-1])*density*binC)

def histvar(binedges,density=None,counts=None,excludeUpperInteger=False,geometric=True,binC = None):
    if(density is None):
        density = normHist(binedges,counts)
    if(binC is None):
        binC = binCenters(binedges,excludeUpperInteger,geometric)
    mean = histmean(binedges,density,binC = binC)
    sqmean = np.nansum((binedges[1:]-binedges[:-1])*density*(binC**2))
    return sqmean - mean**2 

def binnedavg(tbins,times,quantity,geomean=False):
    counts,wcounts,w2counts = binnedavg_accumulate(tbins,times,quantity,geomean=geomean)
    return binnedavg_compute(counts,wcounts,w2counts,geomean=geomean)

def binnedavg_accumulate(tbins,times,quantity,counts=None,wcounts=None,w2counts=None,geomean:bool =False ):
    q_use = quantity 
    if(geomean):
        np.log(quantity)
    counts_temp,_ = np.histogram(times,bins=tbins)
    wcounts_temp,_ = np.histogram(times,bins=tbins,weights=q_use)
    w2counts_temp,_ = np.histogram(times,bins=tbins,weights=q_use**2)
    
    if(not(counts is None)):
        counts+=counts_temp 
        wcounts+=wcounts_temp 
        w2counts+=w2counts_temp 
        return counts,wcounts,w2counts
    else:
        return counts_temp,wcounts_temp,w2counts_temp

def binnedavg_compute(counts,wcounts,w2counts,geomean:bool=False):    
    filt = counts>0

    wcounts[filt]=wcounts[filt]/counts[filt]
    w2counts[filt] = w2counts[filt]/counts[filt]
    std = np.sqrt(w2counts-wcounts**2)
    if(geomean):
        wcounts = np.exp(wcounts)
        std = np.exp(std)

    return wcounts,std

#given a list of measurements (say from different replicates) of q1s and q2 datas, average q2 in terms of q1s. 
def binnedavg_list(q1list,q2list,q1bins=None):
    if(q1bins is None):
        mmax = np.max([np.max(q1) for q1 in q1list])
        mmin = np.min([np.min(q1) for q1 in q1list])
        q1bins = np.linspace(mmin,mmax) 
    q1binc= binCenters(q1bins)#just to get the size right. 
    q1_counts,q1_wcounts,q1_w2counts = np.zeros(shape = q1binc.shape),np.zeros(shape = q1binc.shape),np.zeros(shape = q1binc.shape)
    for q1,q2 in zip(q1list,q2list):
        binnedavg_accumulate(q1bins,q1,q2,q1_counts,q1_wcounts,q1_w2counts)
    q2_q1,q2_q1_std = binnedavg_compute(q1_counts,q1_wcounts,q1_w2counts)
    return q1bins,q2_q1,q2_q1_std

#from a pair of lists of dataset, compute averages in terms of each other? 
def paired_binnedavg(q1list,q2list,q1bins=None,q2bins=None):
    return binnedavg_list(q1list,q2list,q1bins),binnedavg_list(q2list,q1list,q2bins)#(q1bins,q2_q1,q2_q1_std), (q2bins,q1_q2,q1_q2_std) # first, returns q2 as a function of q1, then returns q1 as a function of q2


def plot_paired(strains,stresses,src=plt,skipplot=False, **kwargs):
    #assume structure as:
    xs = np.zeros(strains.size)
    ys = np.zeros(stresses.size)
    xs[::2] = strains[:,0]
    xs[1::2] = strains[:,1]
    ys[::2] = stresses[:,0]
    ys[1::2] = stresses[:,1]
    if(not skipplot):
        src.plot(xs,ys,**kwargs)
    
    return xs,ys

import matplotlib
def get_color_sequential_colormap(attribute,attribute_list,colormap = 'plasma',cmap_range = (0,1)):
    sorted_list = list(attribute_list)
    ind = sorted_list.index(attribute)
    cmap = colormap 
    if(type(colormap) is str):
        cmap = matplotlib.cm.get_cmap(colormap)
    cmin = cmap_range[0]
    cmax = cmap_range[1]
    return cmap(cmin + (cmax-cmin) * ind / (np.size(attribute_list)-1))


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def labelled_arrow(xy1,xy2, t1,t2, txy1=None,txy2=None,src=plt,arrowprops = {},text_kwargs = {'fontsize':6}):
    plt.annotate(t1,xy2,xy1,**{**text_kwargs,'arrowprops':arrowprops})
    plt.text(xy2[0],xy2[1],t2,**text_kwargs)


def add_inset(text,horizontalalignment='left',ax=matplotlib.pyplot,**kwargs):
    xloc = 0.00
    if(horizontalalignment=='left'):
        xloc = 0.01
    else:
        xloc = 0.98
    return ax.text(xloc,.95,text,transform=ax.transAxes, horizontalalignment=horizontalalignment, **kwargs)

def get_param_array_from_dictionaries(list_of_dictionaries, param_name):
    return np.unique(sorted([dic[param_name] for dic in list_of_dictionaries]))


def draw_brace(ax, span, position, text, text_pos, brace_scale=1.0, beta_scale=300., rotate=False, rotate_text=False):
    '''
        all positions and sizes are in axes units
        span: size of the curl
        position: placement of the tip of the curl
        text: label to place somewhere
        text_pos: position for the label
        beta_scale: scaling for the curl, higher makes a smaller radius
        rotate: true rotates to place the curl vertically
        rotate_text: true rotates the text vertically        
    '''
    def rotate_point(x, y, angle_rad):
        cos,sin = np.cos(angle_rad),np.sin(angle_rad)
        return cos*x-sin*y,sin*x+cos*y

    # get the total width to help scale the figure
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    resolution = int(span/xax_span*100)*2+1 # guaranteed uneven
    beta = beta_scale/xax_span # the higher this is, the smaller the radius
    # center the shape at (0, 0)
    x = np.linspace(-span/2., span/2., resolution)
    # calculate the shape
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    # put the tip of the curl at (0, 0)
    max_y = np.max(y)    
    min_y = np.min(y)
    y /= (max_y-min_y)
    y *= brace_scale
    y -= max_y
    # rotate the trace before shifting
    if rotate:
        x,y = rotate_point(x, y, np.pi/2)
    # shift to the user's spot   
    x += position[0]        
    y += position[1]
    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1, clip_on=False)
    # put the text
    ax.text(text_pos[0], text_pos[1], text, ha='center', va='bottom', rotation=90 if rotate_text else 0) 

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    import matplotlib as mpl
    print('legend args:' ,mpl.legend._parse_legend_args([ax], *args, **kwargs))
    handles, labels, args, kwargs = mpl.legend._parse_legend_args([ax], *args, **kwargs)
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mpl.legend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handlelength = kwargs.get('handlelength', 2)
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -handlelength)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [mpl.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s // %s = %s, but should be equal to len(row_labels) = %s." % (len(handles), ncol, nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s // %s = %s, but should be equal to len(row_labels) = %s." % (len(handles), ncol, nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mpl.legend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_


#stolen from: https://stackoverflow.com/questions/55501860/how-to-put-multiple-colormap-patches-in-a-matplotlib-legend
#call with: plt.legend(handles=cmap_handles,             labels=cmap_labels,            handler_map=handler_map,  fontsize=12)

from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, cmap_range=(0,1), **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.cmap_range = cmap_range
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
                          width / self.num_stripes, 
                          height, 
                          fc=self.cmap(self.cmap_range[0] + (self.cmap_range[1]-self.cmap_range[0])*(i) / (self.num_stripes-1)), 
                          transform=trans)
            stripes.append(s)
        return stripes
    def build_legend(handlers,labels,src=plt,**kw):
        #surrogate rectangles.
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in handlers]
        handler_map =  dict(zip(cmap_handles, handlers))
        return src.legend(handles=cmap_handles,handler_map=handler_map,labels=labels,**kw)