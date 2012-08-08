package nc

import (
	"fmt"
	"io"
	"log"
	"runtime"
)

func PrintInfo(out io.Writer) {
	fmt.Fprintln(out, log.Prefix(), "Nimble Cube", runtime.Version(), runtime.Compiler, runtime.GOOS, runtime.GOARCH)
}
