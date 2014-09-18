package engine

// Management of output directory.

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
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
func InitIO(inputfile string, force bool) {
	if outputdir != "" {
		panic("output directory already set")
	}

	InputFile = inputfile
	outputdir = util.NoExt(InputFile) + ".out/"
	LogOut("output directory:", outputdir)

	od := OD()
	if force {
		httpfs.Remove(od)
	}
	if err := httpfs.Mkdir(od); err != nil {
		util.FatalErr(err)
	}

	initLog()
}
