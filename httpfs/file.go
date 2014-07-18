package httpfs

import (
	"fmt"
	"io"
)

// A https File implements a subset of os.File's methods.
type File struct {
	r    io.ReadCloser
	w    io.WriteCloser
	name string
}

func (f *File) Read(p []byte) (n int, err error) {
	if f.r == nil {
		return 0, fmt.Errorf("httpfs: read", f.name, ": write-only file")
	}
	return f.r.Read(p)
}

func (f *File) Write(p []byte) (n int, err error) {
	if f.w == nil {
		return 0, fmt.Errorf("httpfs: write", f.name, ": read-only file")
	}
	return f.w.Write(p)
}

func (f *File) Close() error {
	if f.r != nil {
		return f.r.Close()
	} else {
		return f.w.Close()
	}
}
