//go:build hip

package engine

import (
	"fmt"
	"runtime"

	"github.com/mumax/3/cuda/cu"
)

var UNAME = fmt.Sprintf("%s [%s_%s %s(%s) HIP-%d.%d]",
	VERSION, runtime.GOOS, runtime.GOARCH, runtime.Version(), runtime.Compiler,
	cu.HIP_VERSION/10000000, (cu.HIP_VERSION%10000000)/100000)
