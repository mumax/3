/*
	Go bindings for nVIDIA CUDA 4.
	This package compiles with both gc and gccgo.	
*/
package cuda5

// Dummy imports so that
// 	go get github.com/barnex/cuda5
// will install everything.
import (
	_ "github.com/barnex/cuda5/cu"
	_ "github.com/barnex/cuda5/cufft"
	_ "github.com/barnex/cuda5/safe"
)
