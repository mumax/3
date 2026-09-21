//go:build !hip

package engine

import (
	"fmt"
	"runtime"

	"github.com/mumax/3/cuda/cu"
)

var UNAME = fmt.Sprintf("%s [%s_%s %s(%s) CUDA-%d.%d]",
	VERSION, runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.Compiler,
	cu.CUDA_VERSION/1000, (cu.CUDA_VERSION%1000)/10)
