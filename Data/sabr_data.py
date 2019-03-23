
import json
import os
import numpy as np
from pysabr import hagan_2002_lognormal_sabr as sabrModel






################################################################################################################################
#   Methods definition
################################################################################################################################

def createRandomInputs(strikes, forwards, inputs):
    vals = np.zeros([len(strikes), len(forwards), len(inputs.keys())], dtype=np.float)
    i = 0
    for key in inputs.keys():
        vals[:,:,i] = (inputs[key]['max']-inputs[key]['min'])*np.random.rand(len(forwards), len(strikes)) + inputs[key]['min']
        i=i+1
    return vals


def createSABRVolatilites(strikes, forwards, sabrParams, shift=0.015):
    def sabrSLNvol(k,f): 
        k=int(k)
        f=int(f)
        return sabrModel.lognormal_vol(strikes[k] + shift, forwards[f] + shift, 5, sabrParams[k,f,0], sabrParams[k,f,1], sabrParams[k,f,2], sabrParams[k,f,3])
    return np.fromfunction(np.vectorize(sabrSLNvol), (len(strikes), len(forwards)), dtype=np.float)


def saveImageRBG(filename, data):
    

################################################################################################################################
#   Data generation
################################################################################################################################


try:
	os.chdir(os.path.join(os.getcwd(), 'Data'))
	print(os.getcwd())
except:
	pass

#Open configuration file
with open('data_config.json') as json_file:
    config = json.load(json_file)

numberOfSimulations = 10000

shift = 0.015
strikes = np.arange(config['strikeRng']['min'], config['strikeRng']['max'], config['strikeRng']['step'])
forwards = np.arange(config['forwardRng']['min'], config['forwardRng']['max'], config['forwardRng']['step'])

inputs = dict((k, config[k]) for k in ('alphaRng', 'betaRng', 'rhoRng', 'nuRng'))
sabrParams = createRandomInputs(strikes, forwards, inputs)
sabrVols = createSABRVolatilites(strikes, forwards, sabrParams)
