package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var OD = "." // Output directory

// Sets the output directory where auto-saved files will be stored.
func SetOD(od string, force bool) {
	if OD != "." {
		Fatal(fmt.Errorf("output directory already set to " + OD))
	}
	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	Log("output directory:", OD)

	{ // make OD
		wd, err := os.Getwd()
		Fatal(err)
		stat, err2 := os.Stat(wd)
		Fatal(err2)
		LogErr(os.Mkdir(od, stat.Mode()))
	}

	// fail on non-empty WD
	f, err3 := os.Open(od)
	Fatal(err3)
	files, _ := f.Readdir(1)
	if !force && len(files) != 0 {
		Fatalf(od + " not empty, clean it or force with -f")
	}

	// clean output dir
	if len(files) != 0 && OD != "." {
		Log("cleaning files in", OD)
		filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
			if path != OD {
				Fatal(os.RemoveAll(path))
			}
			return nil
		})
	}
}
