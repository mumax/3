#! /bin/bash
go build test4.go || exit 2
for CPU in 1 4; do
	export GOMAXPROCS=$CPU;
	for N in 9999999 4096 1024 256 128; do
		./test4 -maxblocklen $N || exit 2
	done
done
rm -f ./test4
