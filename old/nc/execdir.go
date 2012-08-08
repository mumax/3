package nc

import (
	"os"
	"path"
)

// Gets the directory where the executable is located.
func ExecutableDir() string {
	exec, err := os.Readlink("/proc/self/exe")
	PanicErr(err)
	return path.Dir(exec)
}
