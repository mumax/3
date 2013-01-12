package core

// Utility functions

import (
	"flag"
	"os"
	"path"
	"strconv"
)

// Open file for writing, error is fatal.
func OpenFile(fname string) *os.File {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	Fatal(err)
	return f
}

// Open file for reading, error is fatal.
func Open(fname string) *os.File {
	f, err := os.Open(fname)
	Fatal(err)
	return f
}

// Remove extension from file name.
func NoExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}

// Product of elements.
func Prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}

// Panics if a != b
func CheckEqualSize(a, b [3]int) {
	if a != b {
		Panic("Size mismatch:", a, "!=", b)
	}
}

func CheckUnits(a, b string) {
	if a != b {
		Panicf(`Unit mismatch: "%v" != "%v"`, a, b)
	}
}

// IntArg returns the idx-th command line as an integer.
func IntArg(idx int) int {
	val, err := strconv.Atoi(flag.Arg(idx))
	Fatal(err)
	return val
}

func Min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func Max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
