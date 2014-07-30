package engine

// Management of output directory.

import (
	"github.com/mumax/3/util"
	"os"
	"path/filepath"
	"strings"
)

var OD = "./" // Output directory

// SetOD sets the output directory where auto-saved files will be stored.
// The -o flag can also be used for this purpose.
func SetOD(od string, force bool) {
	if OD != "./" {
		LogErr("setod: output directory already set to " + OD)
	}

	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	LogOut("output directory:", OD)

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
	defer f.Close()
	files, _ := f.Readdir(1)
	if !force && len(files) != 0 {
		util.Fatal(od, " not empty, clean it or force with -f")
	}

	// clean output dir
	if len(files) != 0 && OD != "." {

		logfile.Close() // Suggested by Raffaele Pellicelli <raffaele.pellicelli@fis.unipr.it> for
		logfile = nil   // windows platform, which cannot remove open file.

		filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
			if path != OD {
				util.FatalErr(os.RemoveAll(path), "clean output directory:")
			}
			return nil
		})
	}
}
