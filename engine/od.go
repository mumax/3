package engine

// Management of output directory.

import (
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

var (
	outputdir string // Output directory
	InputFile string
	fs        *httpfs.Client // abstract file system
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
	if !strings.HasPrefix(inputfile, "http://") {
		mountLocalFS()
		InputFile = inputfile
	} else {
		URL, err := url.Parse(inputfile)
		split := strings.Split(URL.Path, "/")
		if len(split) < 3 {
			util.Fatal("invalid url:", inputfile)
		}
		baseHandler := "/" + split[1]
		InputFile = URL.Path[len(baseHandler):]
		util.Log("inputFile:", InputFile)
		util.Log("dialing:", URL.Scheme+"://"+URL.Host+baseHandler)
		fs, err = httpfs.Dial(URL.Scheme + "://" + URL.Host + baseHandler)
		util.FatalErr(err)
	}

	outputdir = util.NoExt(InputFile) + ".out/"
	LogOut("output directory:", outputdir)

	od := OD()
	{ // make OD
		if err := fs.Mkdir(od, 0777); err != nil && !os.IsExist(err) {
			util.FatalErr(err)
		}
	}

	// fail on non-empty OD

	if force {
		f, err3 := fs.Open(od)
		util.FatalErr(err3)
		defer f.Close()
		files, _ := f.Readdir(1)
		//if !force && len(files) != 0 {
		//	util.Fatal(od, " not empty, clean it or force with -f")
		//}

		// clean output dir
		if len(files) != 0 {
			for _, f := range files {
				fs.RemoveAll(f.Name())
			}
			filepath.Walk(OD(), func(path string, i os.FileInfo, err error) error {
				if path != OD() {
					util.FatalErr(fs.RemoveAll(path))
				}
				return nil
			})
		}
	}

	initLog()
}

// if no (possibly remote) httpfs filesystem is connected,
// mount the local FS.
func mountLocalFS() {
	if fs != nil {
		panic("httpfs already mounted")
	}
	l, err := net.Listen("tcp", ":")
	util.Log("httpfs listening", l.Addr())
	util.FatalErr(err)
	root, eRoot := os.Getwd()
	util.FatalErr(eRoot)
	go func() { util.FatalErr(httpfs.Serve(root, l, "/fs/")) }()
	util.FatalErr(os.Chdir("/")) // avoid accidental use of os instead of fs. TODO: rm
	URL := "http://" + l.Addr().String() + "/fs/"
	fs, err = httpfs.Dial(URL)
	util.FatalErr(err)
	util.Log("connected to httpfs", URL)
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
