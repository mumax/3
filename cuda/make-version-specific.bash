#! /bin/bash

if ((BASH_VERSINFO[0] < 4)); then echo "Sorry, you need at least bash-4.0 to run this script." >&2; exit 1; fi

declare -A CC2PATH

CC2PATH=( [20]=/usr/local/cuda-5.5 \
          [30]=/usr/local/cuda-6.0 \
          [35]=/usr/local/cuda-6.5 \
          [50]=/usr/local/cuda-7.5 \
          [52]=/usr/local/cuda-7.5 \
          [53]=/usr/local/cuda-7.5 \
          [60]=/usr/local/cuda-8.0 \
          [61]=/usr/local/cuda-8.0 \
          [62]=/usr/local/cuda-8.0 \
          [70]=/usr/local/cuda-9.0 \
        )

go build cuda2go.go || exit 1

NVCC='/bin/nvcc --compiler-options -Werror --compiler-options -Wall -Xptxas -O3 -ptx'

for f in *.cu; do
	g=$(echo $f | sed 's/\.cu$//') # file basename
	for cc in ${!CC2PATH[@]}; do
		if [[ $f -nt $g'_'$cc.ptx ]]; then
			echo ${CC2PATH[${cc}]}/$NVCC -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx
			${CC2PATH[${cc}]}/$NVCC -I${CC2PATH[${cc}]}/include -gencode arch=compute_$cc,code=sm_$cc $f -o $g'_'$cc.ptx # error can be ignored
		fi
	done
	if [[ $f -nt $g'_wrapper.go' ]]; then
		./cuda2go $f || exit 1
	fi
done

gofmt -l -s -w *wrapper.go
