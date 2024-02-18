import numpy as np
import scipy
import sys
pyomd_dir = '../../../PyOpenMD'  #change this to the path of the "PyOpenMD" directory
sys.path.append(pyomd_dir)
import pyomd_static
import MDAnalysis as mda
from MDAnalysis import transformations
import copy
np.set_printoptions(threshold=sys.maxsize)

#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

######################## change these parameters ###############################
#input data files ".tpr" and ".xtc" or ".tpr" and ".trr"
#the system topology is in the ".tpr" file
gro = mda.Universe('PE_100c_120ns_slab_solution_npt_sample_1.tpr', 'PE_100c_120ns_slab_solution_npt_sample_1.xtc')
#copying the Universe
gro_copy = gro.copy()

#reference atoms
atoms_ref = gro.select_atoms('name C*')
#copying reference atoms to recover atoms_ref during the loop of the time frames
atoms_ref_copy = gro_copy.select_atoms('name C*')

#selecting compound
#"selec_atoms" keeps the same index order from the xtc/gro files by default
polymer_chains = gro.select_atoms('segid seg_0_S1P1')

#reference axis (z-axis)
axis_ref = np.array([0,0,1], dtype=np.float32)

#threshold radius (same unit as the xyz coordinates)
threshold_radius = 10.8  #in Angstrom, since MDAnalysis convert xyz coordinates to Angstrom

#number of cores
number_of_cores = 24

#output filename
#keep the extension ".xyz.gz"
output_filename = "PE_100c_120ns_slab_P2_order_param_sample_1.xyz.gz"

#from the selected atoms, choose the frequency of the atom that represents the
#monomer
#check "backbones.append(chain.select_atoms('name C1'))" below to change this
#parameter
monomer_C_length = 1
######################## change these parameters ###############################

#wrap trajectories (translating the atoms to the unit cell)
ag = gro.atoms
transform = mda.transformations.wrap(ag)
gro.trajectory.add_transformations(transform)

#wrting the new atomic names and their xyz coordinates to be visualized in OVITO Basic
#"available free of charge under an open source license" (https://www.ovito.org/)
#https://www.ovito.org/docs/current/installation.html
#gz compressed file can be imported directly to OVITO
with mda.coordinates.XYZ.XYZWriter(output_filename, atoms_ref.n_atoms) as W:
    for ts in mda.lib.log.ProgressBar(gro.trajectory, verbose=True, total=len(gro.trajectory)):

        #box unitcell dimensions: [lx, ly, lz, alpha, beta, gamma]
        box_unit_cell = gro.dimensions

        #list of reference atoms
        atoms_reference_list = []
        #list of monomers' indices
        atoms_index_list = []

        for atom_i in atoms_ref:
            #get atomic positions
            atoms_reference_list.append((atom_i.position,axis_ref))
            #get the indices of the reference monomers
            atoms_index_list.append(atom_i.ix)

        #converting python list to numpy array
        atoms_index_list = np.asarray(atoms_index_list)

        #get the polymer chains (fragments)
        chains = polymer_chains.atoms.fragments
        backbones = []

        #list of monomers
        monomers_chain_list = []
        #list of monomers length
        monomers_length_list = []

        for chain in chains:
            ##################### change this parameters #######################
            backbones.append(chain.select_atoms('name C1'))
            ##################### change this parameters #######################

        for chain_i in backbones:
            #get reference monomers (i.e., atoms that represent the monomers)
            monomers_chain_list.extend(chain_i.positions)
            #get the indices of the reference monomers
            monomers_length_list.append(len(chain_i))

        #converting python list to numpy array
        monomers_chain_list = np.asarray(monomers_chain_list)

        #computing the local nematic order parameter for each reference atom
        index_eigs_Q_matrix = pyomd_static.compute_P2_order_parameter_parallel(monomers_chain_list,monomer_C_length,monomers_length_list,
        atoms_reference_list,atoms_index_list,box_unit_cell,num_cores=number_of_cores,use_threshold_radius_=True,threshold_radius_=threshold_radius)

        print("#Time frame =", gro.trajectory.time, "ps;","Number of reference =", len(atoms_ref), "atoms;","Number of chains =",len(chains))
        print("#Atomic index    P2 parameter    Number of unit vectors")
        for rm in range(index_eigs_Q_matrix.shape[0]):
            print('%d' %index_eigs_Q_matrix[rm,0],'    %.3f' %index_eigs_Q_matrix[rm,1],'    %d' %index_eigs_Q_matrix[rm,2])

        #updating the name of the atoms according to their local nematic order value
        atoms_ref_tmp = pyomd_static.update_atomic_names_P2_order_parameters(atoms_ref.names,index_eigs_Q_matrix)

        try:
            #the XYZ writer looks for the atom elements first, instead of atom names
            #see the function def _get_atoms_elements_or_names(self, atoms) at
            #https://github.com/MDAnalysis/mdanalysis/blob/734314b8b7b70617a92eb505f75844e6837505ca/package/MDAnalysis/coordinates/XYZ.py
            atoms_ref.atoms.atoms.elements = atoms_ref_tmp
            W.write(atoms_ref)
            #returning the original element names for the next frame
            atoms_ref.atoms.atoms.elements = atoms_ref_copy.atoms.atoms.elements
        except:
            atoms_ref.names = atoms_ref_tmp
            W.write(atoms_ref)
            #returning the original atomic names for the next frame
            atoms_ref.names = atoms_ref_copy.names
