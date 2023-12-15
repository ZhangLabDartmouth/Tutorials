import numpy as np
import scipy
#from joblib import Parallel, delayed
import sys
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

#Python helper functions to compute static and dynamic properties

################################################################################
##Function to find the minimum vector between 2 coordinates (points) in a
#orthogonal periodic box
##The input parameters are:
#input: vectors_ (numpy array) and ortho_pbc_box_ (numpy array) in the form
#[lx, ly, lz, alpha, beta, gamma] or [lx, ly, lz]
##The ouput parameter is:
#minimum vector "vec_min" (numpy array)
##Minimum image convention for a orthogonal box (ortho_pbc_box_)
#Reference: Allen and Tildesley; "Computer Simulation of Liquids", 2nd ed,
#p. 40--45
##This is a vectorized function to speed up parallel implementations.
##Check "MDAnalysis.lib.distances.minimize_vectors" (package/MDAnalysis/lib/
#distances.py in https://github.com/MDAnalysis/mdanalysis) for triclinic boxes.

def minimize_vectors_pbc(vectors_,ortho_pbc_box_):
    ortho_pbc_box_ = ortho_pbc_box_[:3]
    #convert vectors to a box scale
    pos_in_box = vectors_ / ortho_pbc_box_
    #convert vectors to the unit cell
    vec_min = ortho_pbc_box_ * (pos_in_box - np.rint(pos_in_box))

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
