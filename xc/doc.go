/*
	This package implements the magnetostatic convolution
	on the GPU, transferring input/output to/from the host.	
	Transfers are done asynchronously, i.e., The GPU work
	is overlapped with data transfers.

	TODO: kernel parts could be kept on host when GPU memory is full.
	TODO: check how much memory is used by fft plans, use only one plan?
	TODO: use additional kernel symmetry
	TODO: get rid of conv.realBuf: use 2 small buffers, copyHtoD while copyPadding other.

	Author: Arne Vansteenkiste
*/
package xc
