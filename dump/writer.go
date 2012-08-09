package dump

import (
	"encoding/binary"
	"hash"
	"hash/crc64"
	"io"
	"math"
)

var table = crc64.MakeTable(crc64.ISO)

type Writer struct {
	Header
	out io.Writer
	crc hash.Hash
}

func NewWriter(out io.Writer) *Writer {
	w := new(Writer)
	w.crc = crc64.New(table)
	w.out = io.MultiWriter(w.crc, out)
	return w
}

func (w *Writer) WriteHeader() {

	w.writeString(w.TimeLabel)
	w.writeFloat64(w.Time)
	w.writeString(w.SpaceLabel)
	for _, c := range w.CellSize {
		w.writeFloat64(c)
	}
	//CellSize   [3]float64
	//Rank       int
	//Size       []int
	//Precission int64

}

func (w *Writer) count(n int, err error) {

}

func (w *Writer) writeFloat64(x float64) {
	var buf [8]byte
	binary.PutUvarint(buf[:], math.Float64bits(w.Time))
	w.count(w.out.Write(buf[:]))
}

func (w *Writer) writeString(x string) {

}

func (w *Writer) WriteData(data []float32) error {
	return nil
}

func (w *Writer) WriteHash() error {
	return nil
}
