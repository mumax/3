#! /bin/bash

go build convolution.go || exit 2
for x in 1 ; do #2 3 4 5 8 16; do
	for y in 2 3 4 5 7 8 16 32 64 128 256; do
		for z in 2 3 4 5 9 31 32 33 511 512 513; do
			echo $x $y $z
			./convolution -debug=false $x $y $z || exit 2
		done
	done
done
