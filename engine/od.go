package engine

// Management of output directory.

import (
	"github.com/mumax/3/httpfs"
	"strings"
)

var (
	outputdir string // Output directory
	InputFile string
)

func OD() string {
	if outputdir == "" {
		panic("output not yet initialized")
	}
	return outputdir
}

// SetOD sets the output directory where auto-saved files will be stored.
// The -o flag can also be used for this purpose.
func InitIO(inputfile, od string, force bool) {
	if outputdir != "" {
		panic("output directory already set")
	}

	InputFile = inputfile
	if !strings.HasSuffix(od, "/") {
		od += "/"
	}
	outputdir = od
	if strings.HasPrefix(outputdir, "http://") {
		httpfs.SetWD(outputdir + "/../")
	}
	LogOut("output directory:", outputdir)

	if force {
		httpfs.Remove(od)
	}

	_ = httpfs.Mkdir(od)

	initLog()
}
