package cu

// This file provides CGO flags.

import "C"

//#cgo LDFLAGS:-L/usr/lib/nvidia/ -L/usr/lib64/nvidia/ -L/usr/lib/nvidia-current/ -L/opt/cuda/lib -L/opt/cuda/lib64 -lcuda
//#cgo CFLAGS:-I/usr/local/cuda/include/ -I/opt/cuda/include
import "C"
