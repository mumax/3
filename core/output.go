package core

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

var OD = "." // Output directory

// Sets the output directory where auto-saved files will be stored.
func SetOD(od string) {
	if OD != "." {
		Fatal(fmt.Errorf("output directory already set to " + OD))
	}
	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	Log("output directory:", OD)

	// make output dir
	wd, err := os.Getwd()
	Fatal(err)
	stat, err2 := os.Stat(wd)
	Fatal(err2)
	LogErr(os.Mkdir(od, stat.Mode()))

	f, err3 := os.Open(od)
	Fatal(err3)
	files, err4 := f.Readdir(1)
	Fatal(err4)
	if len(files) != 0 {
		Fatalf(od + " not empty")
	}

	// clean output dir
	filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
		if path != OD {
			Fatal(os.RemoveAll(path))
		}
		return nil
	})
}
