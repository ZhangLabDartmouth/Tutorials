import numpy as np
import scipy
import MDAnalysis as mda
#from joblib import Parallel, delayed
import sys
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

#Python helper functions to compute static and dynamic properties

################################################################################
##Function to find the minimum vector between 2 coordinates (points) in a
#periodic box (orthogonal or triclinic)
##The input parameters are:
#input: vectors_ (numpy array) and pbc_box_ (numpy array) in the form
#[lx, ly, lz, alpha, beta, gamma] (orthogonal or triclinic) or
#[lx, ly, lz] (orthogonal)
##The ouput parameter is:
#minimum vector "vec_min" (numpy array)
##Minimum image convention for a orthogonal box (pbc_box_)
#Reference: Allen and Tildesley; "Computer Simulation of Liquids", 2nd ed,
#p. 40--45
##This is a vectorized function to speed up parallel implementations.
##Check "MDAnalysis.lib.distances.minimize_vectors" (package/MDAnalysis/lib/
#distances.py in https://github.com/MDAnalysis/mdanalysis) for triclinic boxes.

def minimize_vectors_pbc(vectors_,pbc_box_):

    #pbc_box_ with the same precision as the inputted vectors
    pbc_box_ = pbc_box_.astype(vectors_.dtype)

    if np.all(pbc_box_[3:] == 90.) or pbc_box_.shape[0] == 3:
        pbc_box_ = pbc_box_[:3]
        #convert vectors to a box scale
        pos_in_box = vectors_ / pbc_box_
        #convert vectors to the unit cell
        vec_min = pbc_box_ * (pos_in_box - np.rint(pos_in_box))
    else:
        #convert the triclinic box to a matrix representation
        #pbc_box_ = mda.lib.mdamath.triclinic_vectors(pbc_box_)
        #ouput format
        #vec_min = np.empty_like(vectors_)

        #mda.lib.c_distances._minimize_vectors_triclinic(vectors_, pbc_box_.ravel(), vec_min)
        vec_min = mda.lib.distances.minimize_vectors(vectors_,pbc_box_)

    return vec_min
################################################################################


################################################################################
##Functional to fit exponential decay
##The input parameters are:
#x: x-coordinates
#a: constant term that is varied to fit the function
def expfunc(x, a):
    return np.exp(-x/a)
################################################################################
