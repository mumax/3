/*
	Go bindings for nVIDIA CUDA 4.
	This package compiles with both gc and gccgo.	
*/
package cuda4

// Dummy imports so that
// 	go get github.com/barnex/cuda4
// will install everything.
import (
	_ "github.com/barnex/cuda4/cu"
	_ "github.com/barnex/cuda4/cufft"
)
