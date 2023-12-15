#Professor Zhang's Group, Dartmouth College
#https://www.wzdartmouth.com/research

This tutorial computes the local nematic order parameter q. It is defined as
1.5 \lambda_max (eigenvalue) of the ordering matrix Q.
There are two versions: calculation on a single frame and on all the frames.

The data files are available on (Google Drive or Dropbox).

You will need the "PyOpenMD" directory.

After downloading the directory, run the command:

python (or python3) local_nematic_order_parallel_single_frame.py > local_nematic_order_parallel_single_frame.out

The file "local_nematic_order_parallel_single_frame.out" has the local nematic order
for each atomic index. A compressed xyz file is generated (".xyz.gz"). The dynamics
of the local nematic order can be visualized using OVITO Basic.
OVITO Basic "is available free of charge under an open source license" (https://www.ovito.org/)
https://www.ovito.org/docs/current/installation.html

The gz compressed file can be imported directly to OVITO. If the last frame is not
imported, you will need to uncompress the gz file (gzip -dk file.gz) and remove the
last blank line in the xyz file. After removing the blank line, compress it again
(gzip -c filename > filename.gz).


#OVITO dafault legend (in tutorials_pyomd/local_nematic_order_q/ovito_legend)
#Nematic Atom       #q=1.5*\lambda_max       #color
      X                 [0.9, 1.0]           #ffffff
      Y                 [0.8, 0.9[           #67988e
      Z                 [0.7, 0.8[           #33ffff
      A                 [0.6, 0.7[           #b300ff
      D                 [0.5, 0.6[           #ccffb3
      E                 [0.4, 0.5[           #66ff33
      G                 [0.3, 0.4[           #ff66ff
      J                 [0.2, 0.3[           #ffff00
      L                 [0.1, 0.2[           #ff6666
      M                 [0.0, 0.1[           #6666ff
