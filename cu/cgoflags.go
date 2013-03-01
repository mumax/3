package cu

// This file provides CGO flags.

//#cgo LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/lib64/nvidia/ -L/usr/lib/nvidia/ -L/usr/lib/nvidia-current/ -L/opt/cuda/lib64 -L/opt/cuda/lib -lcuda
//#cgo CFLAGS:-I/usr/local/cuda/include/ -I/opt/cuda/include
import "C"
