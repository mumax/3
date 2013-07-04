package cu

// This file provides CGO flags to find CUDA libraries and headers.

////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//
////default location if not properly symlinked:
//#cgo LDFLAGS:-L/usr/local/cuda-5.0/lib64 -L/usr/local/cuda-5.0/lib
//
////ubuntu provided driver:
//#cgo LDFLAGS:-L/usr/lib -L/usr/lib/nvidia-current/ -L/usr/lib/nvidia-experimental -L/usr/lib/nvidia-304 -L/usr/lib/nvidia-310 -L/usr/lib/nvidia-313 -L/usr/lib/nvidia-319
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//
//#cgo LDFLAGS:-lcuda
//
//#cgo CFLAGS:-I/usr/local/cuda/include/ -I/opt/cuda/include
//
////WINDOWS:
//
//#cgo LDFLAGS:-LC:/cuda/v5.0/lib/x64 
//#cgo CFLAGS:-IC:/cuda/v5.0/include 
import "C"
