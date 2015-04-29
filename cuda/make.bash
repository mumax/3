#! /bin/bash

go build cuda2go.go || exit 1

NVCC='nvcc --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx'

for f in *.cu; do
	g=$(echo $f | sed 's/\.cu$//') # file basename
	for cc in 20 30 35 50 52; do
		if [[ $f -nt $g'_'$cc.ptx ]]; then
			echo $NVCC -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx
			$NVCC -I/usr/local/cuda/include -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx # error can be ignored
		fi
	done
	if [[ $f -nt $g'_wrapper.go' ]]; then
		./cuda2go $f || exit 1
	fi
done

