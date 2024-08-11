import numpy as np 
class param_manager:
    def __init__(self, param_ranges):
        self.N_PARAMS = len(param_ranges)
        self.REP_TOTAL = np.product([np.size(prange) for prange in param_ranges])
        self.param_ranges = param_ranges

    def get_params(self,repno):
        reduced = repno 
        params = [] 
        for i,param_range in enumerate(self.param_ranges):
            params.append(param_range[reduced % np.size(param_range)])
            reduced //= np.size(param_range)
        return params 