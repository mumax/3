/*
 Experimental finite difference time domain simulation package.

 CUDA configuration:
 if CUDA is installed in a default location like /usr/local/cuda
 it will probably be found without any configuration.
 Otherwise, add these to your environment:
 	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib
 	export CGO_LDFLAGS='-L<path-to>/cuda/lib64 -L<path-to>/cuda/lib -L/<path-to-libcuda.so> -lcuda -lcufft -lcublas'
 	export CGO_CFLAGS='-I<path-to>/cuda/include'
 where you replace <path-to> by the relevant path.
*/
package main

// dummy imports to make go get fetch all of them.
import (
	_ "code.google.com/p/nimble-cube/dump"
	_ "code.google.com/p/nimble-cube/gpu"
	_ "code.google.com/p/nimble-cube/gpu/ptx"
	_ "code.google.com/p/nimble-cube/mag"
	_ "code.google.com/p/nimble-cube/nimble"
)

func main() {

}
