/*
 GPU magnetostatic convolution with input/output on host.	
 Transfers are done asynchronously, i.e., The GPU work
 is overlapped with data transfers.

 TODO: mv package -> demag_gpu
 TODO: use only one FFT plan, cufft plans use huge buffer space.
 TODO: kernel parts could be kept on host when GPU memory is full.
 TODO: do not keep kernel in host ram if on GPU
 TODO: check how much memory is used by fft plans, use only one plan?
 TODO: use additional kernel symmetry
 TODO: get rid of conv.realBuf: use 2 small buffers, copyHtoD while copyPadding other.

 note: This panic:
 	unexpected fault address 0x***
 	throw: fault
 	[signal 0xb code=0x80 addr=0x*** pc=0x************]
 	...
 	github.com/barnex/cuda4/cu._Cfunc_cuLaunchKernel	
 	...

 means a kernel was launched with the wrong number of arguments.

	Author: Arne Vansteenkiste
*/
package xc
