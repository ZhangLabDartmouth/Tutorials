#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

This tutorial places a PE nanoplastic inside a POPC lipid bilayer (between the
top and bottom leaflets) and adds two water slabs along the z-axis, close to the
leaflets.

You will need the python3 and MDAnalysis (to load the topology and trajectory
files)
https://www.mdanalysis.org/pages/installation_quick_start/

The maximum and minimum x and y dimensions of the lipid bilayer and the water
slab are 15.012 nm and 0.0 nm, respectively. The lipid bilayer comes from
CHARMM-GUI "Membrane Builder" and the water slab from this GROMACS command:

"gmx solvate -cs spc216.gro -o water.gro -box 15.012 15.012 6"


We add a buffer of 2 angstroms between the slabs to facilitate periodic boundary
conditions. Read the section "Periodic Boundary Conditions" at
https://m3g.github.io/packmol/userguide.shtml for more information.


(command line)

python (or python3) merge_nanoplastic_bilayer_water_slabs.py

It will generate the "PE_inside_bilayer.gro" file. During the equilibration steps,
increase the compressibility or the pressure along the z-axis to pack the slabs.
Return to the desired compressibility as the final equilibration step.
