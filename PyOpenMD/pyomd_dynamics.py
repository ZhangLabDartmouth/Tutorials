import numpy as np
import scipy
#import pyomd_math
#import MDAnalysis as mda
from joblib import Parallel, delayed
import sys
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

#Python functions to compute dynamic properties

################################################################################
##Function to compute the end-to-end vector for each chain
##The input parameters are:
#unwrapped "chain_coordinates_" (numpy array): the cartesian coordinates of the
#backbone "monomer_C_length_" (integer number): the frequency that the reference
#atom represents a monomer in the backbone
#Unwrapped coordinates imply that the molecules don't split in different unit cells
##The ouput parameter is:
#"vecs" (numpy array): end-to-end vector
def end_to_end_vector(chain_coordinates_,monomer_C_length_):
    #in case monomer_C_length_ is not inputted as an integer
    monomer_C_length_ = np.int64(monomer_C_length_)

    #end-to-end vector
    vecs = chain_coordinates_[-1] - chain_coordinates_[0]
    #Because the end-to-end vector can be larger than half of the smallest box
    #dimension, pyomd_math.minimize_vectors_pbc or mda.lib.distances.minimize_vectors
    #is not used (otherwise the vector would be from the chain with its own
    #image, i.e., another chain). Instead, the coordinates of fragments (polymers)
    #are unwrapped before calling this function.

    ##"monomer_C_length_" is not required for this function, but it was kept
    #for cases where other atoms are present in the backbone and not only the
    #reference atom
    #the next two lines is another way to compute the end-to-end vector

    #vecs = chain_coordinates_[monomer_C_length_::monomer_C_length_] - chain_coordinates_[:-monomer_C_length_:monomer_C_length_]
    #vecs = vecs.sum(axis=0)

    return vecs
################################################################################


################################################################################
##Function to compute the time autocorrelation function
##The input parameters are:
#"vectors_per_frame_" (numpy array): the cartesian coordinates of the vector
#which will be used in the autocorrelation. The format of this vector should be
#[vectors (time=0), vectors (time=1), vectors (time2), ...], and, for each
#vector, we keep the same order of the chains
#"number_chains_" (integer number): total number of chains in the simulation or
#total number of chains considered (in the case some chains are excluded)
##The ouput parameters are:
#"delta_t" (numpy array): \delta time (x-coordinate)
#"correlation" (numpy array): correlation data (y-coordinate)
def end_to_end_vector_correlation(vectors_per_frame_,number_chains_):
    #in case number_chains_ is not inputted as an integer
    number_chains_ = np.int64(number_chains_)

    #number of frames
    n_frames_ = (vectors_per_frame_.shape[0]/number_chains_)

    ###sanity check
    if n_frames_.is_integer():
        n_frames_ = np.int64(n_frames_)
    else:
        raise ValueError('The number of frames is not an integer.\n'
 'Check the inputed vectors and the number of chains.')
    ###

    #indices of the chain for each time frame
    chain_index = np.array(list(np.arange(np.int64(number_chains_)))*n_frames_)
    #concatenate chain_index to perform the autocorrelation of "chain i" with
    #"chain i"
    index_vectors_per_frame = np.concatenate((chain_index.reshape(-1,1),vectors_per_frame_),axis=1)

    #calculate average vector
    avg_vector = np.mean(vectors_per_frame_,axis=0)

    #Correlation of vectors
    #each column represents a single chain and
    #each row represents a \delta time
    correlation = np.zeros((n_frames_,number_chains_))

    #\delta time
    delta_t = np.arange(n_frames_)

    #number of terms per \delta t
    count_terms = delta_t + 1
    count_terms = count_terms[::-1]

    #Correlation of vectors (upper triangle of a matrix) per chain
    #Take n = delta t, then
    #principal (1st) diagonal (n=0), next upper (2nd) diagonal (n=1),
    #(3rd) diagonal (n=2), ...
    for ij in range(number_chains_):
        vecs_tmp = index_vectors_per_frame[np.int64(index_vectors_per_frame[:,0])==ij]
        correlation_tmp = vecs_tmp[:,1:] @ vecs_tmp[:,1:].T

        for imm in range(correlation_tmp.shape[0]):
            for ikk in range(imm,correlation_tmp.shape[0]):
                correlation[ikk-imm,ij] += correlation_tmp[imm,ikk]

        #average per time count (number of points in a specific delta time)
        correlation[:,ij] = correlation[:,ij] / count_terms

    #average per chain
    correlation = np.mean(correlation - avg_vector @ avg_vector,axis=1)
    #correlation = np.mean(correlation,axis=1) - avg_vector @ avg_vector
    #normalization
    correlation = correlation / correlation[0]

    return delta_t, correlation
################################################################################


################################################################################
##Function to compute the time autocorrelation function
##The input parameters are:
#"vectors_per_frame_" (numpy array): the cartesian coordinates of the vector
#which will be used in the autocorrelation. The format of this vector should be
#[vectors (time=0), vectors (time=1), vectors (time2), ...], and, for each
#vector, we keep the same order of the chains
#"number_chains_" (integer number): total number of chains in the simulation or
#total number of chains considered (in the case some chains are excluded)
##The ouput parameters are:
#"delta_t" (numpy array): \delta time (x-coordinate)
#"correlation" (numpy array): correlation data (y-coordinate)
def end_to_end_vector_correlation_parallel(vectors_per_frame_,number_chains_,num_cores_=-1):
    #in case number_chains_ is not inputted as an integer
    number_chains_ = np.int64(number_chains_)

    #number of frames
    n_frames_ = (vectors_per_frame_.shape[0]/number_chains_)

    ###sanity check
    if n_frames_.is_integer():
        n_frames_ = np.int64(n_frames_)
    else:
        raise ValueError('The number of frames is not an integer.\n'
 'Check the inputed vectors and the number of chains.')
    ###

    #indices of the chain for each time frame
    chain_index = np.array(list(np.arange(np.int64(number_chains_)))*n_frames_)
    #concatenate chain_index to perform the autocorrelation of "chain i" with
    #"chain i"
    index_vectors_per_frame = np.concatenate((chain_index.reshape(-1,1),vectors_per_frame_),axis=1)

    #calculate average vector
    avg_vector = np.mean(vectors_per_frame_,axis=0)

    #Correlation of vectors
    #each column represents a single chain and
    #each row represents a \delta time
    correlation = np.zeros((n_frames_,number_chains_))

    #\delta time
    delta_t = np.arange(n_frames_)

    #number of terms per \delta t
    count_terms = delta_t  + 1
    count_terms = count_terms[::-1]

    #Correlation of vectors (upper triangle of a matrix) per chain
    #Take n = delta t, then
    #principal (1st) diagonal (n=0), next upper (2nd) diagonal (n=1),
    #(3rd) diagonal (n=2), ...
    #for ij in range(number_chains_):
    #    vecs_tmp = index_vectors_per_frame[np.int64(index_vectors_per_frame[:,0])==ij]
    #    correlation_tmp = vecs_tmp[:,1:] @ vecs_tmp[:,1:].T

    #    for imm in range(correlation_tmp.shape[0]):
    #        for ikk in range(imm,correlation_tmp.shape[0]):
    #            correlation[ikk-imm,ij] += correlation_tmp[imm,ikk]

        #average per time count (number of points in a specific delta time)
    #    correlation[:,ij] = correlation[:,ij] / count_terms

    #####parallel version of the uncommented loop above
    def loop_over_chains(ij):
        vecs_tmp = index_vectors_per_frame[np.int64(index_vectors_per_frame[:,0])==ij]
        correlation_tmp = vecs_tmp[:,1:] @ vecs_tmp[:,1:].T

        correlation_loop = np.zeros(n_frames_)

        for imm in range(correlation_tmp.shape[0]):
            for ikk in range(imm,correlation_tmp.shape[0]):
                correlation_loop[ikk-imm] += correlation_tmp[imm,ikk]

        #average per time count (number of points in a specific delta time)
        correlation_loop = correlation_loop / count_terms

        return correlation_loop
    ######

    num_cores = num_cores_
    #print("num_cores=",num_cores)
    tmp_list = Parallel(n_jobs=num_cores)(delayed(loop_over_chains)(ii)
        for ii in range(number_chains_))

    #tmp_list store the data in the same order of the loop
    for kj in range(len(tmp_list)):
        correlation[:,kj] += tmp_list[kj]

    #delete "tmp_list" variable after assignment
    del(tmp_list)

    #average per chain
    correlation = np.mean(correlation - avg_vector @ avg_vector,axis=1)
    #normalization
    correlation = correlation / correlation[0]

    return delta_t, correlation
################################################################################
