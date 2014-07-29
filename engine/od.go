package engine

// Management of output directory.

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"os"
	"path/filepath"
	"strings"
)

var (
	OD = "./"         // Output directory
	fs *httpfs.Client // abstract file system
)

// SetOD sets the output directory where auto-saved files will be stored.
// The -o flag can also be used for this purpose.
func SetOD(od string, force bool) {
	initFS()
	if OD != "./" {
		LogErr("setod: output directory already set to " + OD)
	}

	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	LogOut("output directory:", OD)

	{ // make OD
		_ = fs.Mkdir(od, 0777) // already exists is OK
	}

	// fail on non-empty OD
	f, err3 := fs.Open(od)
	util.FatalErr(err3, "open output directory:")
	files, _ := f.Readdir(1)
	if !force && len(files) != 0 {
		util.Fatal(od, " not empty, clean it or force with -f")
	}

	// clean output dir
	if len(files) != 0 && OD != "." {

		logfile.Close() // Suggested by Raffaele Pellicelli <raffaele.pellicelli@fis.unipr.it> for
		logfile = nil   // windows platform, which cannot remove open file.

		for _, f := range files {
			fs.RemoveAll(f.Name())
		}
		filepath.Walk(OD, func(path string, i os.FileInfo, err error) error {
			if path != OD {
				util.FatalErr(fs.RemoveAll(path), "clean output directory:")
			}
			return nil
		})
	}
}

func initFS() {
	if fs != nil {
		return
	}

}
