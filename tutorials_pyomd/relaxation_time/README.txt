#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

This tutorial computes the relaxation time for iPP18 (isotactic polypropylene with 18 monomers).
There are 64 chains of iPP18 in the simulation.
There are also two additional examples: BS_data and PE_and_seed.

You will need the "PyOpenMD" directory.

After downloading the directory, run the command:

(serial version)

python (or python3) iPP18_relaxation_time_tutorial.py > iPP18_relaxation_time_tutorial.out

(parallel version)

python (or python3) iPP18_relaxation_time_parallel_tutorial.py > iPP18_relaxation_time_parallel_tutorial.out

Compare your results with "iPP18_relaxation_time_tutorial_data.out" or "iPP18_relaxation_time_parallel_tutorial_data.out",
and with the graph "relaxation_time_graph_data".

You can also plot the data in "iPP18_relaxation_time_tutorial.out" and perform a curve fitting using xmgrace:

xmgrace -nxy iPP18_relaxation_time_tutorial.out
