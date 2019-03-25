
import json
import os
import numpy as np
import pandas as pd
from pysabr import hagan_2002_lognormal_sabr as sabrModel
from joblib import Parallel, delayed
import multiprocessing


try:
	os.chdir(os.path.join(os.getcwd(), 'Data'))
	print(os.getcwd())
except:
	pass

#Open configuration file
with open('data_config.json') as json_file:
    config = json.load(json_file)

numberOfSimulations = int(config['numberOfSimulations'])
shift = float(config['shift'])
maturities = [float(x) for x in config['optionMaturities']]
strikes = np.arange(config['strikeRng']['min'], config['strikeRng']['max'], config['strikeRng']['step'])
forwards = np.arange(config['forwardRng']['min'], config['forwardRng']['max'], config['forwardRng']['step'])
inputs = dict((k, config[k]) for k in ('alphaRng', 'betaRng', 'rhoRng', 'nuRng'))
outputNames = ['strike', 'forward', 'maturity', 'alpha', 'beta', 'rho', 'nu', 'sigma']
alreadyGenerated = True

################################################################################################################################
#   Methods definition
################################################################################################################################

def createRandomInputs(strikes, forwards, inputs):
    vals = np.zeros([len(inputs.keys()), len(strikes), len(forwards)], dtype=np.float)
    i = 0
    for key in inputs.keys():
        vals[i,:,:] = (inputs[key]['max']-inputs[key]['min'])*np.random.rand(len(strikes), len(forwards)) + inputs[key]['min']
        i=i+1
    return vals


def createSABRVolatilites(strikes, forwards, maturity, sabrParams, shift=0.015):
    def sabrSLNvol(k,f): 
        k=int(k)
        f=int(f)
        return sabrModel.lognormal_vol(strikes[k] + shift, forwards[f] + shift, maturity, sabrParams[0,k,f], 
                                       sabrParams[1,k,f], sabrParams[2,k,f], sabrParams[3,k,f])
    #Vectorization of calculus
    return np.fromfunction(np.vectorize(sabrSLNvol), (len(strikes), len(forwards)), dtype=np.float)

  
    
def generateSimulationData(nbSimul, strikes, forwards, maturities, shift, inputs, outputNames):
    def oneSample(maturity, inputs):
        nF=len(forwards)
        nS=len(strikes)
        try:
            s = np.repeat(strikes, nF).reshape((nS, nF))
            f = np.repeat(forwards, nS).reshape((nF, nS)).transpose()
            t = np.repeat(maturity, nS*nF).reshape((nS, nF))
            sabrParams = createRandomInputs(strikes, forwards, inputs)
            sabrVols = createSABRVolatilites(strikes, forwards, maturity, sabrParams, shift)
            return np.array([s, f, t, sabrParams[0,], sabrParams[1,], sabrParams[2,], sabrParams[3,], sabrVols]).flatten('F').reshape(nS*nF, 8)
        except:
            return np.zeros((1,8))
    
    #Parallel simulation
    results = np.zeros((1,8))
    for maturity in maturities:
        num_cores = multiprocessing.cpu_count()
        temp = Parallel(n_jobs=num_cores)(delayed(oneSample)(maturity, inputs) for i in range(nbSimul))
        for i in range(len(temp)):
            results = np.append(results, temp[i], axis=0)
    return  pd.DataFrame(np.delete(results, 0, 0), columns=outputNames)

def returnSabrSimulData():
    if( not alreadyGenerated ):
        df = generateSimulationData(numberOfSimulations, strikes, forwards, maturities, shift, inputs, outputNames)
        df.to_csv('sabr_shifted_lognormal_simul.csv')
    else:
        df = pd.read_csv('sabr_shifted_lognormal_simul.csv')
    return df

################################################################################################################################
#   Data generation
################################################################################################################################




