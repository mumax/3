#! /bin/bash

go build cuda2go.go || exit 1

NVCC='nvcc -std c++03 --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx'

for f in *.cu; do

	g=$(echo $f | sed 's/\.cu$//') # file basename

	if [[ $f -nt $g'_wrapper.go' ]]; then

		for cc in 30 35 37 50 52 53 60 61 70 75; do
			if [[ $f -nt $g'_'$cc.ptx ]]; then
				echo $NVCC -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx
				$NVCC -I/usr/local/cuda/include -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx # error can be ignored
			fi
		done

		./cuda2go $f || exit 1
		gofmt -w $g'_wrapper.go'

		rm *.ptx
	fi
done

