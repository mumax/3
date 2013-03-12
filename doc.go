/*
Mumax-cubed (mx³) is a GPU-accelerated finite difference micromagnetic simulation package.

Hardware requirements: you need a CUDA-capable NVIDIA GPU with compute capability 2.0 or higher. An nvidia-provided driver has to be used. Ubuntu users can run:
 	sudo apt-get install nvidia-current
Or the driver can be downloaded from nvidia's website.


Software requirements: for running the pre-compiled mx3 binary, you need install the CUDA toolkit. Ubuntu users can run:
 	sudo apt-get install nvidia-cuda-toolkit
Or the toolkit can be downloaded from nvidia's website.


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
