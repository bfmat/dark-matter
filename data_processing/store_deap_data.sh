#!/bin/sh
# A script to build and run the C++ file for loading and saving the contents of ROOT files containing the DEAP data
# Created by Brendon Matusch, August 2018

# First, compile the code into an object file
g++ -c -I$HOME/ratcage/root-5.34.36/include -I$HOME/ratcage/rat/include -I$HOME/ratcage/local/include/Geant4/ -Wall -g -O2 -march=core2 -mfpmath=sse -fPIC  store_deap_data.cpp
# Next, link it with all of the relevant libraries
g++ -g -O2 -march=core2 -mfpmath=sse -L$HOME/ratcage/root-5.34.36/lib -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -lm -ldl -rdynamic -lm -ldl -o store_deap_data store_deap_data.o -L$HOME/ratcage/root-5.34.36/lib -lGui -lCore -lCint -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -pthread -lm -ldl -rdynamic -lm -ldl  -L/usr/X11R6/lib -lXpm -lX11 -L$HOME/ratcage/rat/lib -lRATEvent
# Run the executable binary
./store_deap_data
# Finally, delete all of the compiled binary files
rm store_deap_data
rm store_deap_data.o
