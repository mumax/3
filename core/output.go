package core

import (
	"os"
	"strings"
)

var (
	OD string // output directory
)

// Sets the output directory where auto-saved files will be stored.
func SetOD(od string) {
	OD = od
	if !strings.HasSuffix(OD, "/") {
		OD += "/"
	}
	Log("output directory:", OD)
	wd, err := os.Getwd()
	Fatal(err)
	stat, err2 := os.Stat(wd)
	Fatal(err2)
	LogErr(os.Mkdir(od, stat.Mode()))
}
