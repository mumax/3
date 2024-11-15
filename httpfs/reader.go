package httpfs

// Utility functions on top of standard httpfs protocol

import (
	"bufio"
	"bytes"
	"io"
)

const BUFSIZE = 16 * 1024 * 1024 // bufio buffer size

// create a file for writing, clobbers previous content if any.
func Create(URL string) (WriteCloseFlusher, error) {
	_ = Remove(URL)
	err := Touch(URL)
	if err != nil {
		return nil, err
	}
	return &bufWriter{bufio.NewWriterSize(&appendWriter{URL, 0}, BUFSIZE)}, nil
}

func MustCreate(URL string) WriteCloseFlusher {
	f, err := Create(URL)
	if err != nil {
		panic(err)
	}
	return f
}

type WriteCloseFlusher interface {
	io.WriteCloser
	Flush() error
}

// open a file for reading
func Open(URL string) (io.ReadCloser, error) {
	data, err := Read(URL)
	if err != nil {
		return nil, err
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

func MustOpen(URL string) io.ReadCloser {
	f, err := Open(URL)
	if err != nil {
		panic(err)
	}
	return f
}

type bufWriter struct {
	buf *bufio.Writer
}

func (w *bufWriter) Write(p []byte) (int, error) { return w.buf.Write(p) }
func (w *bufWriter) Close() error {
	err := w.buf.Flush()
	w.buf = nil // Dangling pointer somewhere?
	if err != nil {
		return err
	}
	return nil
}
func (w *bufWriter) Flush() error { return w.buf.Flush() }

type appendWriter struct {
	URL       string
	byteCount int64
}

func (w *appendWriter) Write(p []byte) (int, error) {
	err := AppendSize(w.URL, p, w.byteCount)
	if err != nil {
		return 0, err // don't know how many bytes written
	}
	w.byteCount += int64(len(p))
	return len(p), nil
}
