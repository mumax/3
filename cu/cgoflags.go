package cu

// This file provides CGO flags to find CUDA libraries and headers.

////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//
////default location if not properly symlinked:
//#cgo LDFLAGS:-L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib
//
////ubuntu provided driver:
//#cgo LDFLAGS:-L/usr/lib/nvidia-current/ -L/usr/lib/nvidia-experimental
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//
////this one is for Mykola and other optimus victims:
//#cgo LDFLAGS:-L/usr/lib64/nvidia/ -L/usr/lib/nvidia/
//
//#cgo LDFLAGS:-lcuda
//
//#cgo CFLAGS:-I/usr/local/cuda/include/ -I/opt/cuda/include
import "C"
