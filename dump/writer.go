package dump

import (
	"hash"
	"hash/crc64"
	"io"
	"math"
	"unsafe"
)

var table = crc64.MakeTable(crc64.ISO)

const (
	CRC_ENABLED  = true
	CRC_DISABLED = false
)

// Writes data frames in dump format.
// Usage:
// 	w := NewWriter(out, CRC_ENABLED)
// 	// set desired w.Header fields
// 	w.WriteHeader()
// 	w.WriteData(array) // may be repeated
// 	w.WriteHash()      // closes this frame.
// 	// may be repeated.
type Writer struct {
	Header       // Written by WriteHeader().
	Bytes  int64 // Total number of bytes written.
	Err    error // Stores the latest I/O error, if any.
	out    io.Writer
	crc    hash.Hash64
}

func NewWriter(out io.Writer, enableCRC bool) *Writer {
	w := new(Writer)
	if enableCRC {
		w.crc = crc64.New(table)
		w.out = io.MultiWriter(w.crc, out)
	} else {
		w.out = out
	}
	return w
}

// Writes the current header.
func (w *Writer) WriteHeader() {
	w.writeString(MAGIC)
	w.writeUInt64(uint64(w.Components))
	for _, s := range w.MeshSize {
		w.writeUInt64(uint64(s))
	}
	for _, s := range w.MeshStep {
		w.writeFloat64(s)
	}
	w.writeString(w.MeshUnit)
	w.writeFloat64(w.Time)
	w.writeString(w.TimeUnit)
	w.writeString(w.DataLabel)
	w.writeString(w.DataUnit)
	w.writeUInt64(FLOAT32) // TODO: check.
}

// Writes the data.
func (w *Writer) WriteData(list []float32) {
	w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)]))
}

// Writes the accumulated hash of this frame, closing the frame.
func (w *Writer) WriteHash() {
	if w.crc == nil {
		w.writeUInt64(0)
	} else {
		w.writeUInt64(w.crc.Sum64())
		w.crc.Reset()
	}
}

func (w *Writer) count(n int, err error) {
	w.Bytes += int64(n)
	if err != nil {
		w.Err = err
	}
}

func (w *Writer) writeFloat64(x float64) {
	w.writeUInt64(math.Float64bits(x))
}

func (w *Writer) writeString(x string) {
	var buf [8]byte
	copy(buf[:], x)
	w.count(w.out.Write(buf[:]))
}

func (w *Writer) writeUInt64(x uint64) {
	w.count(w.out.Write((*(*[8]byte)(unsafe.Pointer(&x)))[:8]))
}
