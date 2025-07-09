package dump

import (
	"bufio"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"os"
	"unsafe"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Write the slice to out in binary format. Add time stamp.
func Write(out io.Writer, s *data.Slice, info data.Meta) error {
	w := newWriter(out)

	// Writes the header.
	w.writeString(MAGIC)
	w.writeUInt64(uint64(s.NComp()))
	size := s.Size()
	w.writeUInt64(uint64(size[2])) // backwards compatible coordinates!
	w.writeUInt64(uint64(size[1]))
	w.writeUInt64(uint64(size[0]))
	cell := info.CellSize
	w.writeFloat64(cell[2])
	w.writeFloat64(cell[1])
	w.writeFloat64(cell[0])
	w.writeString(info.MeshUnit)
	w.writeFloat64(info.Time)
	w.writeString("s") // time unit
	w.writeString(info.Name)
	w.writeString(info.Unit)
	w.writeUInt64(4) // precision

	// return header write error before writing data
	if w.err != nil {
		return w.err
	}

	w.writeData(s)
	w.writeHash()
	return w.err
}

// Write the slice to file in binary format. Add time stamp.
func WriteFile(fname string, s *data.Slice, info data.Meta) error {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	return Write(w, s, info)
}

// Write the slice to file in binary format, panic on error.
func MustWriteFile(fname string, s *data.Slice, info data.Meta) {
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

// Writes the data.
func (w *writer) writeData(array *data.Slice) {
	data := array.Tensors()
	size := array.Size()

	ncomp := array.NComp()
	for c := 0; c < ncomp; c++ {
		for iz := 0; iz < size[2]; iz++ {
			for iy := 0; iy < size[1]; iy++ {
				for ix := 0; ix < size[0]; ix++ {
					w.writeFloat32(data[c][iz][iy][ix])
				}
			}
		}
	}
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

func (w *writer) writeFloat32(x float32) {
	var bytes []byte
	bytes = (*[4]byte)(unsafe.Pointer(&x))[:]
	w.count(w.out.Write(bytes))
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

// product of elements.
func prod(size [3]int) int {
	return size[0] * size[1] * size[2]
}
