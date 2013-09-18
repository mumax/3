package data

import (
	"code.google.com/p/mx3/util"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"os"
	"unsafe"
)

// Write the slice to out in binary format. Add time stamp.
func Write(out io.Writer, s *Slice, info Meta) error {
	w := newWriter(out)

	// Writes the header.
	w.writeString(MAGIC)
	w.writeUInt64(uint64(s.NComp()))
	for _, s := range s.Mesh().Size() {
		w.writeUInt64(uint64(s))
	}
	for _, s := range s.Mesh().CellSize() {
		w.writeFloat64(s)
	}
	w.writeString(s.Mesh().Unit)
	w.writeFloat64(info.Time)
	w.writeString("s") // time unit
	w.writeString(info.Name)
	w.writeString(info.Unit)
	w.writeUInt64(4) // precission
	for i := 0; i < padding; i++ {
		w.writeUInt64(0)
	}

	// return header write error before writing data
	if w.err != nil {
		return w.err
	}

	data := s.Host()
	for _, d := range data {
		w.writeData(d)
	}
	w.writeHash()
	return w.err
}

// Write the slice to file in binary format. Add time stamp.
func WriteFile(fname string, s *Slice, info Meta) error {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return err
	}
	defer f.Close()
	return Write(f, s, info)
}

// Write the slice to file in binary format, panic on error.
func MustWriteFile(fname string, s *Slice, info Meta) {
	err := WriteFile(fname, s, info)
	util.FatalErr(err)
}

var table = crc64.MakeTable(crc64.ISO)

type writer struct {
	out io.Writer
	crc hash.Hash64
	err error
}

func newWriter(out io.Writer) *writer {
	w := new(writer)
	w.crc = crc64.New(table)
	w.out = io.MultiWriter(w.crc, out)
	return w
}

const MAGIC = "#dump002" // identifies dump format
const padding = 0        // padding words before data section

// Writes the data.
func (w *writer) writeData(list []float32) {
	w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)]))
}

// Writes the accumulated hash of this frame, closing the frame.
func (w *writer) writeHash() {
	w.writeUInt64(w.crc.Sum64())
	w.crc.Reset()
}

func (w *writer) count(n int, err error) {
	if err != nil && w.err == nil {
		w.err = err
	}
}

func (w *writer) writeFloat64(x float64) {
	w.writeUInt64(math.Float64bits(x))
}

func (w *writer) writeString(x string) {
	var buf [8]byte
	copy(buf[:], x)
	w.count(w.out.Write(buf[:]))
}

func (w *writer) writeUInt64(x uint64) {
	w.count(w.out.Write((*(*[8]byte)(unsafe.Pointer(&x)))[:8]))
}
