package mx

// File: misc utility functions
// Author: Arne Vansteenkiste

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path"
)

// Open file for writing, error is fatal.
func OpenFile(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	FatalErr(err)
	return f
}

// Open file for reading, error is fatal.
func Open(fname string) *os.File {
	f, err := os.Open(fname)
	FatalErr(err)
	return f
}

// Remove extension from file name.
func NoExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}

// Exec command and write output to outfile.
func SaveCmdOutput(outfile string, cmd string, args ...string) {
	Log("exec:", cmd, args, ">", outfile)
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		Fatalf("exec %v %v: %v: %v", cmd, args, err, string(out))
	} else {
		Fatalf("writing %v: %v", outfile, ioutil.WriteFile(outfile, out, 0666))
	}
}

// path to the executable.
func ProcSelfExe() string {
	me, err := os.Readlink("/proc/self/exe")
	PanicErr(err)
	return me
}

// Product of elements.
func Prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Panics if a != b
//func CheckEqualSize(a, b [3]int) {
//	if a != b {
//		Panic("Size mismatch:", a, "!=", b)
//	}
//}

//func CheckUnits(a, b string) {
//	if a != b {
//		Panicf(`Unit mismatch: "%v" != "%v"`, a, b)
//	}
//}

// IntArg returns the idx-th command line as an integer.
//func IntArg(idx int) int {
//	val, err := strconv.Atoi(flag.Arg(idx))
//	Fatal(err)
//	return val
//}

//func Min(x, y int) int {
//	if x < y {
//		return x
//	}
//	return y
//}
//
//func Max(x, y int) int {
//	if x > y {
//		return x
//	}
//	return y
//}
