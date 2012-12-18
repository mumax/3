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
	PanicErr(err) // todo: fatal
	stat, err2 := os.Stat(wd)
	PanicErr(err2) // todo: fatal
	LogErr(os.Mkdir(od, stat.Mode()))

	f, err3 := os.Open(od)
	Fatal(err3)
	files, _ := f.Readdir(1)
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
