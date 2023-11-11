from Data_generator import Model
import matplotlib.pyplot as plt
import numpy as np
from gaussian_density import Gaussian
from monotracker import Monotracker
from scipy.io import savemat
from MultiTracker import Multitracker
def ospa(x, y, cutoff=100.0, p=1.0):
    """
    Optimal Subpattern Assignment (OSPA) distance between two finite sets.

    Parameters
    ----------
    x : ndarray
    y : ndarray
        Finite sets as 2D arrays, where each column is an element.

    cutoff : float
        Cut off parameter.

    p : float
        p-parameter of the metric.

    Returns
    -------
    dist : float
    """

    num_x, num_y = x.shape[1], y.shape[1]

    # if both sets are empty
    if num_x == 0 and num_y == 0:
        return 0

    # if at least one is empty
    if num_x == 0 or num_y == 0:
        return cutoff

    # compute cost matrix
    d = np.sqrt(np.sum((np.tile(x, num_y) - np.repeat(y, num_x, axis=1)) ** 2, axis=0))
    d = d.reshape(num_x, num_y).T
    d = np.minimum(d, cutoff) ** p

    # solve optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(d)
    # compute cost of the optimal assignment
    cost = d[row_ind, col_ind].sum()

    dist = ((cutoff ** p * np.abs(num_y - num_x) + cost) / np.max((num_y, num_x))) ** (1/p)

    return dist

model = Model()
truth =model.gen_truth()
meas = model.gen_meas(truth)
density = Gaussian
#tracker = Monotracker(density, model)
tracker = Multitracker(density, model)
#est = tracker.gaussian_sum_filter(density,model,meas)
#est = tracker.bernoulli_gms_filter(density,model,meas)
est = tracker.cmbm_filter(density,model,meas)
X = truth['X']
K = truth['K']
Z = meas['Z']
xe = est['X']
print(density.esf(np.asarray(Z[1][...,0])))
#savemat('est.mat',est)
#savemat('truth.mat',truth)
for k in range(K):
    plt.figure(1)
    plt.plot(Z[k][0,:],Z[k][1,:],'.',c='b')
    plt.figure(2)
    mask = np.isnan(X[:,k,:])
    no_nan_cols = np.where(np.all(~mask, axis=0))[0]
    x = X[:,k,:][:,no_nan_cols]
    if x.sum():
        plt.plot(x[0,:],x[2,:],'+')
    #print(xe[k])
    if not isinstance(xe[k],type(None)):
        plt.plot(x[0,:],x[2,:],'.',c='r')
plt.show()