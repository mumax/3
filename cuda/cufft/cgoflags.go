package cufft

// This file provides CGO flags to find CUDA libraries and headers.

//#cgo LDFLAGS:-lcufft
//
////default location:
//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//#cgo CFLAGS: -I/usr/local/cuda/include/
//
////Ubuntu 15.04:
//#cgo LDFLAGS:-L/usr/lib/x86_64-linux-gnu/
//#cgo CFLAGS: -I/usr/include
//
////arch linux:
//#cgo LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//#cgo CFLAGS: -I/opt/cuda/include
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/lib/x64
//#cgo windows CFLAGS: -IC:/cuda/include -w
import "C"
