package util

import (
	"path"
)

// Remove extension from file name.
func NoExt(file string) string {
	ext := path.Ext(file)
	return file[:len(file)-len(ext)]
}

// Panics if err is not nil. Signals a bug.
func PanicErr(err error) {
	if err != nil {
		panic(err)
	}
}
