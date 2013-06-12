#! /bin/bash

NVCC='nvcc --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx'

# ! when changing supported compute capabilities, cuda2go.go should be modified (cc list)
for f in *.cu; do
	g=$(basename -s .cu $f)
	for cc in 20 30 35; do
		if [[ $f -nt $g'_'$cc.ptx ]]; then
			echo $NVCC -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx
			$NVCC -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx || exit 1
		fi
	done
done

go build cuda2go.go
./cuda2go *.cu
go fmt
go install -v
astyle *.cu
