package engine

// Management of output directory.

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"net"
	"os"
	"path/filepath"
	"strings"
)

var (
	OD = "/"          // Output directory
	fs *httpfs.Client // abstract file system
)

// SetOD sets the output directory where auto-saved files will be stored.
// The -o flag can also be used for this purpose.
func SetOD(od string, force bool) {
	initFS()
	if OD != "/" {
		LogErr("setod: output directory already set to " + OD)
	}

	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	LogOut("output directory:", OD)

	{ // make OD
		if err := fs.Mkdir(od, 0777); err != nil && !os.IsExist(err) {
			util.FatalErr(err)
		}
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
	l, err := net.Listen("tcp", ":")
	util.Log("httpfs listening", l.Addr())
	util.FatalErr(err)
	root, eRoot := os.Getwd()
	util.FatalErr(eRoot)
	go func() { util.FatalErr(httpfs.Serve(root, l)) }()
	util.FatalErr(os.Chdir("/")) // avoid accidental use of os instead of fs. TODO: rm
	fs, err = httpfs.Dial(l.Addr().String())
	util.FatalErr(err)
	util.Log("connected to httpfs", l.Addr())
}

func MountHTTPFS(addr string) {
	if fs != nil {
		util.Fatal("httpfs already mounted")
	}

	var err error
	fs, err = httpfs.Dial(addr)
	util.FatalErr(err)
	util.Log("connected to httpfs", addr)
}
