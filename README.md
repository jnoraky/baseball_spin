# baseball_spin

Overview
========
We estimate spin by passing in the frames from the video to our algorithm.
The algorithm does a search for the best rotation from which the spin and axis
can be obtained. 

Set-up
======
0. Download the bin files and the frames folder from Dropbox. These files should
   be in the same directory as the source code. 
1. Compile the code e.g. g++ -o out.exe -O3 *.cpp ``pkg-config --cflags --libs opencv``
2. Run it e.g. ./out.exe 3 1000 (here, 3 refers to the third video, and 1000
   refers to the number of spins to try)

Issues
======
* Assets are quite large and slow to load
* Algorithm can be slow depending on the number of rotations/spins to try 
* Change input directly to video instead of frames

