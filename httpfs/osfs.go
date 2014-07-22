package httpfs

import (
	"os"
)

// OSFS resembles client but uses the underlying os file system directly.
type OSFS struct{}

// Delegates to os.Open.
func (f *OSFS) Open(name string) (*os.File, error) {
	return os.Open(name)
}

// Delegates to os.Create.
func (f *OSFS) Create(name string) (*os.File, error) {
	return os.Create(name)
}

// Delegates to os.OpenFile.
func (f *OSFS) OpenFile(name string, flag int, perm os.FileMode) (*os.File, error) {
	return os.OpenFile(name, flag, perm)
}
