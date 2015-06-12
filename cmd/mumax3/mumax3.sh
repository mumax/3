#! /bin/bash
# 
# This script adds the current directory to your library path
# and launches mumax3 using the shipped cuda libraries.
# 
# When you have correctly set-up cuda, you can just run
# mumax directly without this wrapper.
# 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
./mumax3-cuda6.0 $@
