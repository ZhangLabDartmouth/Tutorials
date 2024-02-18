import numpy as np
import scipy
import pyomd_math
#import MDAnalysis as mda
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

#Python functions to compute static properties

################################################################################
##Function to compute the ordering matrix Q (or local nematic order)
#input: unit_tangent_vectors_ (numpy array) are the unit tangent vectors
#used to compute the matrix Q
#return: averaged Q_matrix (symmetric matrix), i.e., we compute a Q matrix
#for each unit_tangent_vectors_ and average its value

def ordering_matrix_Q(unit_tangent_vectors_):

    Q_matrix = 0

    #reshape array in case its dimension is lower than 2
    if unit_tangent_vectors_.ndim < (2 - 1e-3):
        unit_tangent_vectors_ = unit_tangent_vectors_.reshape(1,-1)

    ###check if unit_tangent_vectors_ are unit vectors
    check_unit = scipy.linalg.norm(unit_tangent_vectors_,axis=1).reshape(-1,1)

    #MDAnalysis uses numpy.ndarray with dtype numpy.float32
    if (np.fabs(check_unit - 1.) > 5e-7).any():
        raise ValueError('There are tangent vectors that are not unit vectors.\n'
'Check your unit vectors before computing the ordering matrix Q.')
    ###

    #Q_matrix_tmp = 0
    #for lm in unit_tangent_vectors_:
    #    Q_matrix_tmp += np.outer(lm,lm) - (1.0/3)*np.identity(3)

    Q_matrix = unit_tangent_vectors_.T @ unit_tangent_vectors_ - (1.0/3)*unit_tangent_vectors_.shape[0]*np.identity(3)

    Q_matrix = Q_matrix / (unit_tangent_vectors_.shape[0])

    #checking if Q_matrix is a symmetric matrix (Q_matrix = Q_matrix.T)
    if not np.all(np.abs(Q_matrix-Q_matrix.T) < 1.0e-14):
        print("################################################")
        print("Check if the Q matrix is symmetric")
        print()
        print("Q_matrix = ")
        print(Q_matrix)
        print()
        print("Q_matrix - Q_matrix.T = ")
        print(Q_matrix-Q_matrix.T)
        print()
        print("################################################")

    #eigs_Q_matrix, eigenv_Q_matrix = scipy.linalg.eigh(Q_matrix)

    #if not np.iscomplex(eigs_Q_matrix.all()):
    #    eigs_Q_matrix = eigs_Q_matrix.real

    return Q_matrix #, eigs_Q_matrix, eigenv_Q_matrix
################################################################################


################################################################################
##Function to compute the P2 order parameter
#input: unit_tangent_vectors_ (numpy array) are the unit tangent vectors and
#unit_axis_array_ (numpy array) are the reference axes. Both input should have
#the same dimensions, e.g., n x 3, where n stands for the number of unit tangent
#vectors and 3 for 3D-points. If there is only one reference axis, use the
#np.repeat option to create a array with the same dimension as the unit tangent
#vectors.
#return: averaged P2_parameter (scalar), i.e., we compute a P2 parameter
#for each unit_tangent_vectors_ and average its value

def order_parameter_P2(unit_tangent_vectors_,unit_axis_array_):

    P2_parameter = 0

    #reshape arrays in case their dimensions are lower than 2
    if unit_tangent_vectors_.ndim < (2 - 1e-3):
        unit_tangent_vectors_ = unit_tangent_vectors_.reshape(1,-1)

    if unit_axis_array_.ndim < (2 - 1e-3):
        unit_axis_array_ = unit_axis_array_.reshape(1,-1)

    ###check if unit_tangent_vectors_ are unit vectors
    check_unit_tangent = scipy.linalg.norm(unit_tangent_vectors_,axis=1).reshape(-1,1)
    check_unit_axis = scipy.linalg.norm(unit_axis_array_,axis=1).reshape(-1,1)

    #MDAnalysis uses numpy.ndarray with dtype numpy.float32
    if (np.fabs(check_unit_tangent - 1.) > 5e-7).any():
        raise ValueError('There are tangent vectors that are not unit vectors.\n'
'Check your unit vectors before computing the P2 order parameter.')

    if (np.fabs(check_unit_axis - 1.) > 5e-7).any():
        raise ValueError('There are axes that are not unit vectors.\n'
'Check your unit vectors before computing the P2 order parameter.')
    ###

    #(P2)_k = ( 3. * cos^2(\theta_{ij})_k - 1. ) / 2.
    P2_parameter = ( 3.0 * np.sum(unit_tangent_vectors_ * unit_axis_array_, axis=1)**2 - 1.0 ) / 2.0

    #P2 = sum_k{(P2)_k}
    P2_parameter = P2_parameter.sum() / (unit_tangent_vectors_.shape[0])

    return P2_parameter
################################################################################


################################################################################
##Function to extract the unit tangent vectors around a reference point
#input: chain_coordinates_ (numpy array), monomer_C_length_ is the frequency of
#the atom that represents the monomers, box_ is the box dimension (numpy array)
#in the form [lx, ly, lz, alpha, beta, gamma] or
#[lx, ly, lz] = [Lx, Ly, Lz, 0. , 0. , 0.], interchain_ is a condition to exclude
#the chain of the reference atom (if it is in the chain), use_threshold_radius
#is a condition, reference_point_ is a numpy array and threshold_radius is a
#scalar (same unit as the inputted chain_coordinates_). In MDAnalysis, the
#coordinates are in Angstrom (https://userguide.mdanalysis.org/stable/units.html).
#If use_threshold_radius=False, the function computes all the unit tangent
#vectors for the chain_coordinates_
#return: unit tangent vectors (numpy array)

def unit_vectors_around_reference_point(chain_coordinates_, monomer_C_length_, box_,
interchain=False, use_threshold_radius=True, reference_point_=np.array([0.,0.,0.]), threshold_radius=10.8):
    #in case monomer_C_length_ is not inputted as an integer
    monomer_C_length_ = np.int64(monomer_C_length_)

    if interchain:
        #minimum image convention to vectors (periodic boundary conditions) in
        #the case the coordinates are not wrapped
        minimim_vec = pyomd_math.minimize_vectors_pbc(chain_coordinates_ - reference_point_, box_)
        distance = scipy.linalg.norm(minimim_vec, axis=1)

        if np.min(distance) < 1e-6:
            return list()

    if use_threshold_radius:
        #coordinates of each monomer
        chain_coordinates_ = chain_coordinates_[::monomer_C_length_]

        #index of chain coordinates (a numpy array)
        index_coordinates_ = np.arange(chain_coordinates_.shape[0])

        chain_coordinates_ = np.concatenate((index_coordinates_.reshape(-1,1),chain_coordinates_),axis=1)

        #minimum image convention to vectors (periodic boundary conditions)
        minimim_vec = pyomd_math.minimize_vectors_pbc(chain_coordinates_[:,1:] - reference_point_, box_)

        distance = scipy.linalg.norm(minimim_vec, axis=1)
        index_threshold = (distance <= threshold_radius)
        chain_coordinates_ = chain_coordinates_[index_threshold,:]

        #Vector from each monomer to the next (tangent vector)
        vecs = chain_coordinates_[1::1] - chain_coordinates_[:-1:1]
        #using the minimum image convention to vectors (applying the periodic boundary conditions)
        vecs[:,1:] = pyomd_math.minimize_vectors_pbc(vecs[:,1:], box_)

        #using the difference from indices to choose the tangent vectors around the reference point
        #if the difference between indices (integers) is larger than 1, the vector is not a tangent vector
        index_diff = (vecs[:,0] < 1.5)  #using 1.5 to avoid floating point arithmetic issues
        vecs = vecs[index_diff,:]

        #recovering the tangent vectors
        vecs = vecs[:,1:]

    else:
        #Vector from each monomer to the next (tangent vector)
        vecs = chain_coordinates_[monomer_C_length_::monomer_C_length_] - chain_coordinates_[:-monomer_C_length_:monomer_C_length_]
        #using the minimum image convention to vectors (applying the periodic boundary conditions)
        vecs = pyomd_math.minimize_vectors_pbc(vecs, box_)

    #Normalized to unit vectors
    vecs = vecs / scipy.linalg.norm(vecs,axis=1).reshape(-1,1)

    return vecs
################################################################################


################################################################################
##Function to compute the local nematic order using the ordering matrix Q
#input: chain_list_ is the chain coordinates (numpy array), monomer_C_length_ is
#the frequency of the atom that represents the monomers, monomers_length_list_
#is number of monomers for each chain in chain_list_ (python list),
#atoms_reference_list_ is the reference atoms where the local parameter is
#computed (python list), atoms_index_list_ (numpy array) contains the indices of
#the reference atoms in "atoms_reference_list_", box_ is the box dimension
#(numpy array) in the form [lx, ly, lz, alpha, beta, gamma] or
#[lx, ly, lz] = [Lx, Ly, Lz, 0. , 0. , 0.], num_cores is the number of cores
#used in the calculation (num_cores=-1 uses all the cores in the machine),
#interchain_ is a condition to exclude the chain of the reference atom in the
#calculation, use_threshold_radius is a condition, and threshold_radius is a
#scalar (same unit as the inputted chain_coordinates_). In MDAnalysis, the
#coordinates are in Angstrom (https://userguide.mdanalysis.org/stable/units.html).
#return: atoms_index, 1.5*max_eigval, number_unit_vectors around the reference
#atom

def compute_local_nematic_order_parallel(chain_list_,monomer_C_length_,monomers_length_list_,
atoms_reference_list_,atoms_index_list_,box_,num_cores=-1,interchain_=False,use_threshold_radius_=True,threshold_radius_=10.8):

    ###parallel loop
    def loop_atoms_reference_list(kl):
        list_unit_vectors = []

        length_index = 0

        for im in monomers_length_list_:
            #get unit vectors for each reference monomer (atom)
            list_tmp = unit_vectors_around_reference_point(chain_list_[length_index:length_index+im], monomer_C_length_, box_, interchain=interchain_,
            use_threshold_radius=use_threshold_radius_, reference_point_=kl, threshold_radius=threshold_radius_)
            list_unit_vectors.extend(list_tmp)  #an empty "list_tmp" is automatically not added to the "list_unit_vectors"

            length_index += im

        number_unit_vectors = len(list_unit_vectors)

        if number_unit_vectors > 0:
            list_unit_vectors = np.asarray(list_unit_vectors)
            avg_Q_matrix_frame = ordering_matrix_Q(list_unit_vectors)

            #eigenvalues and eigenvectors of the Q matrix
            eigs_avg_Q_matrix, eigenv_avg_Q_matrix = scipy.linalg.eigh(avg_Q_matrix_frame)

            #index of the max eigenvalue
            index_max = eigs_avg_Q_matrix.argmax()
            #max eigenvalue
            max_eigval = eigs_avg_Q_matrix[index_max]
            #eigenvector associated with the max eigenvalue
            #max_eigval_eigenvector = eigenv_avg_Q_matrix[:,index_max]
        else:
            max_eigval = np.nan

        return 1.5*max_eigval, number_unit_vectors
        ###

    output = list(
    tqdm(Parallel(return_as="generator", n_jobs=num_cores)(delayed(loop_atoms_reference_list)(ii)
        for ii in atoms_reference_list_),
        total=len(atoms_reference_list_)))

    output = np.asarray(output)
    output = np.concatenate((atoms_index_list_.reshape(-1,1),output),axis=1)

    return output  #return atoms_index, 1.5*max_eigval, number_unit_vectors
################################################################################


################################################################################
##Function to compute the local nematic order using the P2 order parameter
#input: chain_list_ (numpy array) is the chain coordinates, monomer_C_length_ is
#the frequency of the atom that represents the monomers, monomers_length_list_
#is number of monomers for each chain in chain_list_ (python list),
#atoms_reference_list_ is the concatenate reference atoms (first three points)
#and reference axes (last three points) where the P2 order parameter is
#computed (python list), atoms_index_list_ (numpy array) contains the indices of
#the reference atoms in "atoms_reference_list_", box_ is the box dimension
#(numpy array) in the form [lx, ly, lz, alpha, beta, gamma] or
#[lx, ly, lz] = [Lx, Ly, Lz, 0. , 0. , 0.], num_cores is the number of cores
#used in the calculation (num_cores=-1 uses all the cores in the machine),
#interchain_ is a condition to exclude the chain of the reference atom in the
#calculation, use_threshold_radius is a condition, and threshold_radius is a
#scalar (same unit as the inputted chain_coordinates_). In MDAnalysis, the
#coordinates are in Angstrom (https://userguide.mdanalysis.org/stable/units.html).
#return: atoms_index, 1.5*max_eigval, number_unit_vectors around the reference
#atom

def compute_P2_order_parameter_parallel(chain_list_,monomer_C_length_,monomers_length_list_,
atoms_reference_list_,atoms_index_list_,box_,num_cores=-1,interchain_=False,use_threshold_radius_=True,threshold_radius_=10.8):

    ###parallel loop
    def loop_atoms_reference_list(kl):
        list_unit_vectors = []

        length_index = 0

        reference_atom_pos = kl[0]
        reference_atom_axis = kl[1]
        #reference_atom_axis = reference_atom_axis.reshape(-1,1)

        for im in monomers_length_list_:
            #get unit vectors for each reference monomer (atom)
            list_tmp = unit_vectors_around_reference_point(chain_list_[length_index:length_index+im], monomer_C_length_, box_, interchain=interchain_,
            use_threshold_radius=use_threshold_radius_, reference_point_=reference_atom_pos, threshold_radius=threshold_radius_)
            list_unit_vectors.extend(list_tmp)  #an empty "list_tmp" is automatically not added to the "list_unit_vectors"

            length_index += im

        number_unit_vectors = len(list_unit_vectors)

        if number_unit_vectors > 0:
            #unit vectors
            list_unit_vectors = np.asarray(list_unit_vectors)

            #reference axis
            unit_axis_array = np.tile(reference_atom_axis, (list_unit_vectors.shape[0],1))
            #Normalize reference axis to unit vectors
            unit_axis_array = unit_axis_array / scipy.linalg.norm(unit_axis_array,axis=1).reshape(-1,1)

            avg_P2_order_param_frame = order_parameter_P2(list_unit_vectors,unit_axis_array)

        else:
            avg_P2_order_param_frame = np.nan

        return avg_P2_order_param_frame, number_unit_vectors
        ###

    output = list(
    tqdm(Parallel(return_as="generator", n_jobs=num_cores)(delayed(loop_atoms_reference_list)(ii)
        for ii in atoms_reference_list_),
        total=len(atoms_reference_list_)))

    output = np.asarray(output)
    output = np.concatenate((atoms_index_list_.reshape(-1,1),output),axis=1)

    return output  #return atoms_index, 1.5*max_eigval, number_unit_vectors
################################################################################


################################################################################
##Function to change the name of atoms according to their local nematic order
#input: atomic_name_ is the atomic names in the same order as the atomic indices
#(numpy array) and indices_nematic_data_ is the data where the atomic indices
#(1st column) and their local nematic order (2nd column) are stored (numpy array)
#return: new atomic names following the legend below
#(atom 'X', 1.5*\lambda_max >= 0.9)
#(atom 'Y', 0.8 <= 1.5*\lambda_max < 0.9)
#(atom 'Z', 0.7 <= 1.5*\lambda_max < 0.8)
#(atom 'A', 0.6 <= 1.5*\lambda_max < 0.7)
#(atom 'D', 0.5 <= 1.5*\lambda_max < 0.6)
#(atom 'E', 0.4 <= 1.5*\lambda_max < 0.5)
#(atom 'G', 0.3 <= 1.5*\lambda_max < 0.4)
#(atom 'J', 0.2 <= 1.5*\lambda_max < 0.3)
#(atom 'L', 0.1 <= 1.5*\lambda_max < 0.2)
#(atom 'M', 1.5*\lambda_max < 0.1)

def update_atomic_names_local_nematic_order(atomic_name_,indices_nematic_data_):
    #selecting local nematic order
    #atom 'X'
    mask_tmp_10 = indices_nematic_data_[:,1] >= 0.9
    #atom 'Y'
    mask_tmp_9 = (indices_nematic_data_[:,1] >= 0.8) & (indices_nematic_data_[:,1] < 0.9)
    #atom 'Z'
    mask_tmp_8 = (indices_nematic_data_[:,1] >= 0.7) & (indices_nematic_data_[:,1] < 0.8)
    #atom 'A'
    mask_tmp_7 = (indices_nematic_data_[:,1] >= 0.6) & (indices_nematic_data_[:,1] < 0.7)
    #atom 'D'
    mask_tmp_6 = (indices_nematic_data_[:,1] >= 0.5) & (indices_nematic_data_[:,1] < 0.6)
    #atom 'E'
    mask_tmp_5 = (indices_nematic_data_[:,1] >= 0.4) & (indices_nematic_data_[:,1] < 0.5)
    #atom 'G'
    mask_tmp_4 = (indices_nematic_data_[:,1] >= 0.3) & (indices_nematic_data_[:,1] < 0.4)
    #atom 'J'
    mask_tmp_3 = (indices_nematic_data_[:,1] >= 0.2) & (indices_nematic_data_[:,1] < 0.3)
    #atom 'L'
    mask_tmp_2 = (indices_nematic_data_[:,1] >= 0.1) & (indices_nematic_data_[:,1] < 0.2)
    #atom 'M'
    mask_tmp_1 = indices_nematic_data_[:,1] < 0.1

    mask_list = [mask_tmp_10,mask_tmp_9,mask_tmp_8,mask_tmp_7,mask_tmp_6,
     mask_tmp_5,mask_tmp_4,mask_tmp_3,mask_tmp_2,mask_tmp_1]

    letters='XYZADEGJLM'

    for imm in range(len(mask_list)):
        atomic_name_[mask_list[imm]] = letters[imm]

    return atomic_name_
################################################################################


################################################################################
##Function to change the name of atoms according to their local nematic order
#input: atomic_name_ is the atomic names in the same order as the atomic indices
#(numpy array) and indices_nematic_data_ is the data where the atomic indices
#(1st column) and their local nematic order (2nd column) are stored (numpy array)
#return: new atomic names following the legend below
#(atom 'X', P2 >= 0.9)
#(atom 'Y', 0.8 <= P2 < 0.9)
#(atom 'Z', 0.7 <= P2 < 0.8)
#(atom 'A', 0.6 <= P2 < 0.7)
#(atom 'D', 0.5 <= P2 < 0.6)
#(atom 'E', 0.4 <= P2 < 0.5)
#(atom 'G', 0.3 <= P2 < 0.4)
#(atom 'J', 0.2 <= P2 < 0.3)
#(atom 'L', 0.1 <= P2 < 0.2)
#(atom 'M', 0.0 <= P2 < 0.1)
#(atom 'Q', -0.1 <= P2 < 0.0)
#(atom 'R', -0.2 <= P2 < -0.1)
#(atom 'T', -0.3 <= P2 < -0.2)
#(atom 'U', -0.4 <= P2 < -0.3)
#(atom 'V', P2 < -0.4)

def update_atomic_names_P2_order_parameters(atomic_name_,indices_nematic_data_):
    #selecting local nematic order
    #atom 'X'
    mask_tmp_15 = indices_nematic_data_[:,1] >= 0.9
    #atom 'Y'
    mask_tmp_14 = (indices_nematic_data_[:,1] >= 0.8) & (indices_nematic_data_[:,1] < 0.9)
    #atom 'Z'
    mask_tmp_13 = (indices_nematic_data_[:,1] >= 0.7) & (indices_nematic_data_[:,1] < 0.8)
    #atom 'A'
    mask_tmp_12 = (indices_nematic_data_[:,1] >= 0.6) & (indices_nematic_data_[:,1] < 0.7)
    #atom 'D'
    mask_tmp_11 = (indices_nematic_data_[:,1] >= 0.5) & (indices_nematic_data_[:,1] < 0.6)
    #atom 'E'
    mask_tmp_10 = (indices_nematic_data_[:,1] >= 0.4) & (indices_nematic_data_[:,1] < 0.5)
    #atom 'G'
    mask_tmp_9 = (indices_nematic_data_[:,1] >= 0.3) & (indices_nematic_data_[:,1] < 0.4)
    #atom 'J'
    mask_tmp_8 = (indices_nematic_data_[:,1] >= 0.2) & (indices_nematic_data_[:,1] < 0.3)
    #atom 'L'
    mask_tmp_7 = (indices_nematic_data_[:,1] >= 0.1) & (indices_nematic_data_[:,1] < 0.2)
    #atom 'M'
    mask_tmp_6 = (indices_nematic_data_[:,1] >= 0.0) & (indices_nematic_data_[:,1] < 0.1)
    #atom 'Q'
    mask_tmp_5 = (indices_nematic_data_[:,1] >= -0.1) & (indices_nematic_data_[:,1] < 0.0)
    #atom 'R'
    mask_tmp_4 = (indices_nematic_data_[:,1] >= -0.2) & (indices_nematic_data_[:,1] < -0.1)
    #atom 'T'
    mask_tmp_3 = (indices_nematic_data_[:,1] >= -0.3) & (indices_nematic_data_[:,1] < -0.2)
    #atom 'U'
    mask_tmp_2 = (indices_nematic_data_[:,1] >= -0.4) & (indices_nematic_data_[:,1] < -0.3)
    #atom 'V'
    mask_tmp_1 = indices_nematic_data_[:,1] < -0.4

    mask_list = [mask_tmp_15,mask_tmp_14,mask_tmp_13,mask_tmp_12,mask_tmp_11,
     mask_tmp_10,mask_tmp_9,mask_tmp_8,mask_tmp_7,mask_tmp_6,
     mask_tmp_5,mask_tmp_4,mask_tmp_3,mask_tmp_2,mask_tmp_1]

    letters='XYZADEGJLMQRTUV'

    for imm in range(len(mask_list)):
        atomic_name_[mask_list[imm]] = letters[imm]

    return atomic_name_
################################################################################
