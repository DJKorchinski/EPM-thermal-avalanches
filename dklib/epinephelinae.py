#Epinephelinae, the subfamily of fish that includes the Grouper

def make_series(data,matching_parameters,vary_params,varying_params_to_numpify=[]):
    series_list = []
    for dic in data:
        found = False
        for series in series_list:
            match = True
            for param in matching_parameters:
                if(not(dic[param] == series[param])):
                    match = False 
                    break 
            if(match):
                found = True
                break 
        if(not found):
            #make a new series:
            series = {  }
            for param in matching_parameters:
                series[param] = dic[param]
            series['data'] = []
            for param in vary_params:
                series[param] = []
            series_list.append(series)
        series['data'].append(dic)
        for param in vary_params:
            series[param].append(dic[param])
    for series in series_list:
        for param in varying_params_to_numpify:
            import numpy as np
            series[param] = np.array(series[param])
    return series_list 