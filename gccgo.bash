#! /bin/bash

go build -o mx3 -v -compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'


