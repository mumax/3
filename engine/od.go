package engine

// Management of output directory.

import (
	"code.google.com/p/mx3/util"
	"log"
	"os"
	"path/filepath"
	"strings"
)

var OD = "./" // Output directory

// SetOD sets the output directory where auto-saved files will be stored.
// The -o flag can also be used for this purpose.
func SetOD(od string, force bool) {
	if OD != "./" {
		log.Println("setod: output directory already set to", OD)
		return
	}

	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	log.Println("output directory:", OD)

	{ // make OD
		wd, err := os.Getwd()
		util.FatalErr(err, "create output directory:")
		stat, err2 := os.Stat(wd)
		util.FatalErr(err2, "create output directory:")
		_ = os.Mkdir(od, stat.Mode()) // already exists is OK
	}

	// fail on non-empty OD
	f, err3 := os.Open(od)
	util.FatalErr(err3, "open output directory:")
	files, _ := f.Readdir(1)
	if !force && len(files) != 0 {
		log.Fatal(od, " not empty, clean it or force with -f")
	}

	// clean output dir
	if len(files) != 0 && OD != "." {
		log.Println("cleaning files in", OD)
		filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
			if path != OD {
				util.FatalErr(os.RemoveAll(path), "clean output directory:")
			}
			return nil
		})
	}
}
