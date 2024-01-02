import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
import sys
np.set_printoptions(threshold=sys.maxsize)


#loading PE nanoplastic surrounded by water
gro = mda.Universe('PE_small_solution_after_annealing.pdb')
#selecting the PE nanoplastic
PE = gro.select_atoms('resname ETHY')

#loading water slab nanoplastic
gro_water_1 = mda.Universe('water.gro')

#wrapping the water slab in its unit cell
ag_water = gro_water_1.atoms
transform_water = mda.transformations.wrap(ag_water)
gro_water_1.trajectory.add_transformations(transform_water)

#copying the water slab
gro_water_2 = gro_water_1.copy()

#selecting all the water in both slabs
water_1 = gro_water_1.select_atoms('all')
water_2 = gro_water_2.select_atoms('all')

#loading the POPC bilayer surrounded by water
gro_bilayer = mda.Universe('step5_input.psf','step5_input.gro')

#wrapping the POPC bilayer slab in its unit cell
ag = gro_bilayer.atoms
transform = mda.transformations.wrap(ag)
gro_bilayer.trajectory.add_transformations(transform)
#selecting the POPC bilayer
POPC_bilayer = gro_bilayer.select_atoms('resname POPC')

#bilayer leaflets
#top leaflet: the first 330 fragments
#POPC_bilayer_leaflet_1 = POPC_bilayer.fragments[:330]
#bottom leaflet: the last 330 fragments
#POPC_bilayer_leaflet_2 = POPC_bilayer.fragments[330:]

#finding the indices of the bilayer leaflets
#for chain_i in POPC_bilayer_leaflet_1:
#    print(chain_i.ix)

#using the previous indices to select the leaflets
POPC_bilayer_leaflet_1 = POPC_bilayer.select_atoms('index 0:44219')
POPC_bilayer_leaflet_2 = POPC_bilayer.select_atoms('index 44220:88439')
#maximum and minimum z-axis of the leaflets
POPC_bilayer_leaflet_1_z_max = np.max(POPC_bilayer_leaflet_1.positions[:,2])
POPC_bilayer_leaflet_1_z_min = np.min(POPC_bilayer_leaflet_1.positions[:,2])

POPC_bilayer_leaflet_2_z_max = np.max(POPC_bilayer_leaflet_2.positions[:,2])
POPC_bilayer_leaflet_2_z_min = np.min(POPC_bilayer_leaflet_2.positions[:,2])

#maximum and minimum z-axis of the PE nanoplastic
PE_z_max = np.max(PE.positions[:,2])
PE_z_min = np.min(PE.positions[:,2])

#opening the POPC lipid bilayer
#top leaflet
POPC_bilayer_leaflet_1.positions = POPC_bilayer_leaflet_1.positions + np.array([0,0,PE_z_max]) - np.array([0,0,POPC_bilayer_leaflet_1_z_min]) + np.array([0,0,2])
#bottom leaflet
POPC_bilayer_leaflet_2.positions = POPC_bilayer_leaflet_2.positions + np.array([0,0,PE_z_min]) - np.array([0,0,POPC_bilayer_leaflet_2_z_max]) + np.array([0,0,-2])

#new maximum and minimum z-axis of the leaflets after the opening
POPC_bilayer_leaflet_1_z_max = np.max(POPC_bilayer_leaflet_1.positions[:,2])
POPC_bilayer_leaflet_1_z_min = np.min(POPC_bilayer_leaflet_1.positions[:,2])

POPC_bilayer_leaflet_2_z_max = np.max(POPC_bilayer_leaflet_2.positions[:,2])
POPC_bilayer_leaflet_2_z_min = np.min(POPC_bilayer_leaflet_2.positions[:,2])

#maximum and minimum z-axis of the water slab
water_z_max = np.max(water_1.positions[:,2])
water_z_min = np.min(water_1.positions[:,2])

#placing the water slabs along the z-axis
water_1.positions = water_1.positions + np.array([0,0,-water_z_min+POPC_bilayer_leaflet_1_z_max]) + np.array([0,0,2])
water_2.positions = water_2.positions + np.array([0,0,-water_z_max+POPC_bilayer_leaflet_2_z_min]) + np.array([0,0,-2])

#combining the PE nanoplastic and the slabs
combined = mda.Merge(PE, POPC_bilayer_leaflet_1, POPC_bilayer_leaflet_2, water_1, water_2)

#maximum and minimum coordinates of the combined (merged) system
combined_coord_max = np.max(combined.atoms.positions,axis=0)
combined_coord_min = np.min(combined.atoms.positions,axis=0)

#similation box dimensions
x_box = combined_coord_max[0] - combined_coord_min[0] + 2
y_box = combined_coord_max[1] - combined_coord_min[1] + 2
z_box = combined_coord_max[2] - combined_coord_min[2] + 2

combined.dimensions = [x_box, y_box, z_box, 90, 90, 90]

combined_system = combined.select_atoms('all')

#moving the minimum coordinate values to a zero reference (vmd will show
#everything in a unit cell)
combined_system.positions = combined_system.positions - np.array([combined_coord_min[0],combined_coord_min[1],combined_coord_min[2]])

with mda.Writer('PE_inside_bilayer.gro') as w:
    w.write(combined)
