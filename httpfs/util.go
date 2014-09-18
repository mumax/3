package httpfs

// Utility functions on top of standard httpfs protocol

import (
	"bytes"
	"io"
	"io/ioutil"
)

// create a file for writing, clobbers previous content if any.
func Create(URL string) (io.WriteCloser, error) {
	_ = Remove(URL)
	err := Touch(URL)
	if err != nil {
		return nil, err
	}
	return &appendWriter{URL}, nil
}

// open a file for reading
func Open(URL string) (io.ReadCloser, error) {
	data, err := Read(URL)
	if err != nil {
		return nil, err
	}
	return ioutil.NopCloser(bytes.NewReader(data)), nil
}

func Touch(URL string) error {
	return Append(URL, []byte{})
}

type appendWriter struct {
	URL string
}

// TODO: buffer heavily, Flush() on close
func (w *appendWriter) Write(p []byte) (int, error) {
	err := Append(w.URL, p)
	if err != nil {
		return 0, err // don't know how many bytes written
	}
	return len(p), nil
}

// TODO: flush
func (w *appendWriter) Close() error {
	return nil
}
