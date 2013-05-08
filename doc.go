/*
Mumax-cubed (mx³) is a GPU-accelerated finite difference micromagnetic simulation package.


GPU hardware and driver

You need a CUDA-capable NVIDIA GPU with compute capability 2.0 or higher. An nvidia-provided driver has to be used. Ubuntu users can run:
 	sudo apt-get install nvidia-current
Or the driver can be downloaded from nvidia's website.


CUDA toolkit

For running the pre-compiled mx3 binary, you need install the CUDA toolkit. Ubuntu users can run:
 	sudo apt-get install nvidia-cuda-toolkit
Or the toolkit can be downloaded from nvidia's website.



Getting started

Running an input script:
 	mx3 myscript.txt

Example script:
 	setgridsize(128,      32,      1)
 	setcellsize(3.125e-9, 3.125e-9, 3e-9)

 	alpha = 0.02
 	msat  = 800e3
 	aex   = 13e-12
 	m     = uniform(1, .1, 0)
 	b_ext = (-24.6E-3, 4.3E-3, 0)

 	savetable(10e-12)
 	autosave(m, 50e-12)

 	run(1e-9)

 	print("final m:", average(m))

See package "examples" for more examples.

Also, more advanced input scripts can be written in Go. See package "engine".



Web interface

While the simulation is running, you can visualize and manipulate it from your browser. Default is:
 	http://localhost:35367
The -http flag can select an other port.


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
