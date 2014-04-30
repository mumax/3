#! /bin/bash
# 
# Build script using gccgo, which may or may not be faster than gc and which may or may not compile...
#
go install -v -compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native' github.com/mumax/3/... 


