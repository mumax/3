package nc

import (
	"fmt"
	"io"
	"runtime"
)

func PrintInfo(out io.Writer) {
	fmt.Fprintln(out, "Nimble Cube", runtime.Version(), runtime.Compiler, runtime.GOOS, runtime.GOARCH)
}
