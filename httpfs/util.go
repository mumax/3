package httpfs

import (
	"io"
)

func Create(URL string) (io.WriteCloser, error) {
	_ = Remove(URL)
	return &appendWriter{URL}
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
