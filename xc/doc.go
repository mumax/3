/*
	This package implements the magnetostatic convolution
	on the GPU, transferring input/output to/from the host.	
	Transfers are done asynchronously, i.e., The GPU work
	is overlapped with data transfers.

	TODO: kernel parts could be kept on host when GPU memory is full.

	Author: Arne Vansteenkiste
*/
package xc
