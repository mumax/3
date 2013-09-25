#! /bin/bash

go build -o mumax3 -v -compiler gccgo -gccgoflags '-static-libgcc -O4 -Ofast -march=native'


