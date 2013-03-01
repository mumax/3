package cufft

// This file provides CGO flags to find CUDA libraries and headers.

////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//
////default location if not properly symlinked:
//#cgo LDFLAGS:-L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//
//#cgo LDFLAGS:-lcudart -lcufft
//
//#cgo CFLAGS:-I/usr/local/cuda/include/ -I/opt/cuda/include
import "C"

