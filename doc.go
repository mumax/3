/*
Mumax-cubed (mx³) is a GPU-accelerated finite difference micromagnetic simulation package.


GPU hardware and driver

You need a CUDA-capable NVIDIA GPU with compute capability 2.0 or higher. An nvidia-provided driver has to be used. Ubuntu users can run:
 	sudo apt-get install nvidia-current
Or the driver can be downloaded from nvidia's website (see below).


CUDA toolkit

For running the pre-compiled mx3 binary, you need install the CUDA toolkit. Ubuntu users can run:
 	sudo apt-get install nvidia-cuda-toolkit
Or the toolkit can be downloaded from nvidia's website https://developer.nvidia.com/cuda-downloads (toolkit download also includes driver).



Getting started

Running an input script:
 	mx3 myscript.txt

Example scripts can be found in the examples/ subdirectory, or on http://godoc.org/code.google.com/p/mx3/examples.
Also, more advanced input scripts can be written in Go. See package "engine".

Scripts can be checked for errors without running them with:
 	mx3 -vet script1.txt script2.txt ...

To find out all mx3 command line flags, run:
 	mx3 -help


Web interface

While the simulation is running, you can visualize and manipulate it from your browser. Default is:
 	http://localhost:35367
The -http flag can select an other port.


Output

All output is saved in a '.out' directory with the same base name as your input file (can be overridden by -o flag). Vector data is stored in an efficient binary 'dump' format that can be converted with the mx3-convert tool. See tools/mx3-convert or http://godoc.org/code.google.com/p/mx3/tools/mx3-convert.


Recompiling

The pre-compiled binaries should suit most people. Nevertheless, if you want to compile yourself you need Git and Go (with properly set $GOPATH). Ubuntu users can run:
	sudo apt-get install git golang
Be sure to set you $GOPATH before proceeding. E.g. in your ~/.bashrc, add something like:
 	export GOPATH=/home/me/gocode

If CUDA is installed in a default location like /usr/local/cuda
it will probably be found without any configuration.
Otherwise, add these to your environment:

 	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib
 	export CGO_LDFLAGS='-L<path-to>/cuda/lib64 -L<path-to>/cuda/lib -L/<path-to-libcuda.so> -lcuda -lcufft'
 	export CGO_CFLAGS='-I<path-to>/cuda/include'

where you replace <path-to> by the relevant path.


*/
package main
