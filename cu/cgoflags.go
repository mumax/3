package cu

// This file provides CGO flags.

import "C"

//#cgo LDFLAGS:-L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia/ -L/usr/lib64/nvidia/ -L/usr/lib/nvidia-current/ -lcuda
//#cgo CFLAGS:-I/usr/local/cuda/include/
import "C"
