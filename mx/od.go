package mx

// File: management of output directory.
// Author: Arne Vansteenkiste

import (
	"os"
	"path/filepath"
	"strings"
)

var OD = "./" // Output directory

// SetOD sets the output directory where auto-saved files will be stored.
func SetOD(od string, force bool) {
	if OD != "./" {
		FatalExit("output directory already set to", OD)
	}
	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	Log("output directory:", OD)

	{ // make OD
		wd, err := os.Getwd()
		FatalErr(err, "create output directory:")
		stat, err2 := os.Stat(wd)
		FatalErr(err2, "create output directory:")
		LogErr(os.Mkdir(od, stat.Mode())) // already exists is OK
	}

	// fail on non-empty OD
	f, err3 := os.Open(od)
	FatalErr(err3, "open output directory:")
	files, _ := f.Readdir(1)
	if !force && len(files) != 0 {
		FatalExit(od, "not empty, clean it or force with -f")
	}

	// clean output dir
	if len(files) != 0 && OD != "." {
		Log("cleaning files in", OD)
		filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
			if path != OD {
				FatalErr(os.RemoveAll(path), "clean output directory:")
			}
			return nil
		})
	}
}

// called by init()
func initOD() {
	if *Flag_od != "" {
		SetOD(*Flag_od, *Flag_force)
	}
}
