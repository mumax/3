package dump

import (
	"hash"
	"hash/crc64"
	"io"
)

var table = crc64.MakeTable(crc64.ISO)

type Writer struct {
	Header
	out io.Writer
	crc hash.Hash
}

func NewWriter(out io.Writer) *Writer {
	w := new(Writer)
	w.out = out
	w.crc = crc64.New(table)
	return w
}

func (w *Writer) WriteHeader() error {
	return nil
}

func (w *Writer) WriteData(data []float32) error {
	return nil
}

func (w *Writer) WriteHash() error {
	return nil
}
