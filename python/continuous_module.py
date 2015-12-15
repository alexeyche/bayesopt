


import numpy as np
import bayesopt as bo

from bayesopt import ContinuousModel



class ConcreteContinuousModel(ContinuousModel):
	def __init__(self, ndim, params):
		ContinuousModel.__init__(self, ndim, params)


	def evaluateSample(self, Xin):
	    total = 5.0
	    for value in Xin:
	        total = total + (value -0.33)*(value-0.33)

	    return total


ndim = 10

params = {}
params['n_iterations'] = 50
params['n_iter_relearn'] = 5
params['n_init_samples'] = 2

ccm = ConcreteContinuousModel(ndim, params)

lb = np.zeros((ndim,))
ub = np.ones((ndim,))

ccm.setBoundingBox(lb, ub)
ccm.optimize()