import numpy as np
import numpy as np
import scipy
import sys
pyomd_dir = '../../../PyOpenMD'  #change this to the path of the "PyOpenMD" directory
sys.path.append(pyomd_dir)
import pyomd_dynamics
import pyomd_math
import MDAnalysis as mda
from MDAnalysis import transformations
#from joblib import Parallel, delayed
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

######################## change these parameters ###############################
#input data files ".tpr" and ".xtc" or ".tpr" and ".trr"
#the system topology is in the ".tpr" file
gro = mda.Universe('prod_0_0.tpr', 'prod_0_0.xtc')

#number of chains (polymers)
n_chains = 161

#number of monomers per chain
n_monomores = 400

#the frequency that the reference atom represents a monomer in the backbone
monomer_C_length = 1  #check the select_atoms below before changing this parameter
######################## change these parameters ###############################

#list of end-to-end vector correlation
list_end_vectors = []
list_time_step = []
number_atoms = 0

#unwrap the data, i.e., molecules are not split over the boundaries of
#the box (periodic boundary condition)
ag = gro.atoms
transform = mda.transformations.unwrap(ag)
gro.trajectory.add_transformations(transform)

#loop over frames
for ts in mda.lib.log.ProgressBar(gro.trajectory, verbose=True,
                          total=len(gro.trajectory)):

    #tuple of chains
    #the first n_chains in "fragments" are the polymer chains
    #use "print(gro.atoms.fragments[0:n_chains])" to check your system
    list_time_step.append(ts.dt) #in picoseconds (ps)
    chains = gro.atoms.fragments[0:n_chains]
    #print(ts.time,ts.dt)

    ########################change this parameters##############################
    #C1 is the reference atom of the monomer
    #backbones returns only the coordinates of atom C1 for each chain
    backbones = [chain.select_atoms('name C1') for chain in chains]
    ########################change this parameters##############################

    #get end-to-end vectors for each chain
    for chain_i in backbones:
        list_end_vectors.append(pyomd_dynamics.end_to_end_vector(chain_i.positions,monomer_C_length))
        number_atoms += chain_i.positions.shape[0]


#converting python list to numpy array
vectors_frame = np.asarray(list_end_vectors) #in Angstrom
time_step = np.asarray(list_time_step) #in picoseconds (ps)

###sanity check
##checking the number of frames, chains and monomers
n_frames = len(gro.trajectory)
if len(list_end_vectors) != (n_frames*n_chains):
    print('Check if the number of chains is correct.\n'
 'Check if the frames were propely compressed (e.g., last frame).')

if (number_atoms) != (n_frames*n_chains*n_monomores):
    print('Check if the number of monomers is correct.\n'
 'Check if the frames were propely compressed (e.g., last frame).')

#checking if the time steps are equal
if not np.all(np.isclose(time_step, time_step[0])):
    print('There are different time steps in the trajectory data.\n'
 'Check your trajectory file.')
###

#computing the time correlation function
delta_time, correlation_function = pyomd_dynamics.end_to_end_vector_correlation_parallel(vectors_frame,n_chains)

#converting delta_time to a time unit using the time step among the time frames
delta_time = delta_time*time_step[0] / (1e3) #now in nanoseconds (ns)

#fitting an exponential decay to the time correlation function
#in correlation functions, there are more points at the beginning of the curve
#and these are statistically more significant
#using the first 30 points for the fitting
#popt, pcov = scipy.optimize.curve_fit(expfunc, delta_time[:30], correlation_function[:30])
#popt, pcov = scipy.optimize.curve_fit(pyomd_math.expfunc, delta_time[:70], correlation_function[:70])
#popt = popt.item() #relaxation time
#perr = (np.sqrt(np.diag(pcov))).item()  #one standard deviation

#printing data
############### uncomment these lines to print the data ########################
print('#Total number of atoms considered in all chains per frame =', (number_atoms / n_frames))
print("#Time frame (ns)   Vector Correlation")
for i in range(delta_time.shape[0]):
    print('%.8f       %.8f' % (delta_time[i],correlation_function[i]))
############### uncomment these lines to print the data ########################

#plotting the time correlation function (all points) and the fitted curve
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(delta_time,correlation_function,'ro',label='Result')
#ax.plot(delta_time,pyomd_math.expfunc(delta_time,popt),label=r'Curve Fit: $\tau=$'+'{:.1f}'.format(popt)+r'$\,\pm\,$'+'{:.1f}'.format(perr)+' ns')
ax.set_xlabel(r'$\Delta t$ (nm)')
ax.set_ylabel(r'$C(\Delta t)$')

ax.legend(loc='best')
#fig.savefig("relaxation_time_graph.pdf")
plt.show()
