import pickle
#loads avalanches one at a time from a source!
class av_loader():
    def __init__(self, fno):
        self.fno = fno 

    def loadnextseg(self):
        try:
            self.dti = pickle.load(self.fh)
            self.avind = 0
            self.seglen = len(self.dti['Avalanches']) 
            return True  
        except (EOFError):
            self.fh.close()
            return False 

    def __iter__(self):
        #initializes loading  
        self.fh = open(self.fno,'rb')
        self.loadnextseg() 
        return self 

    def __next__(self):
        while(self.avind >= self.seglen):
            if(not self.loadnextseg()):
                raise StopIteration
        av = self.dti['Avalanches'][self.avind]
        self.avind+=1 
        return av 


#loads some property, one at a time, and returns that property along with the most recently recorded avalanche
class gen_loader():
    def __init__(self, fno, propname,freq):
        self.fno = fno 
        self.prop = propname
        # The frequency, in avalanches, of the event. 
        # So that, if propname is recorded whenever np.mod(avNum,freq) = 0, we can find the corresponding avalanche
        self.freq = freq 

    def loadnextseg(self):
        try:
            self.dti = pickle.load(self.fh)
            self.propind = 0
            self.seglen = len(self.dti[self.prop])
            self.avsOffset += self.avslen 
            self.avslen = len(self.dti['Avalanches'])
            return True  
        except (EOFError):
            self.fh.close()
            return False 

    def __iter__(self):
        #initializes loading  
        self.fh = open(self.fno,'rb')
        self.avNum = 0
        self.avsOffset = 0
        self.avslen = 0
        self.loadnextseg()
        return self 

    def __next__(self):
        while(self.propind >= self.seglen):
            if(not self.loadnextseg()):
                raise StopIteration
        try:
            av = self.dti['Avalanches'][self.avNum - self.avsOffset]
        except (IndexError):
            print('index error on avs.', self.propind, self.avNum,self.avsOffset, self.freq)
            raise StopIteration
        prop = self.dti[self.prop][self.propind]
        self.avNum += self.freq 
        self.propind+=1
        return prop,av