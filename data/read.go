package data

import (
	"fmt"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"os"
	"unsafe"
)

func ReadSlice(in io.Reader) (*Slice, error) {
	r := newReader(in)
	return r.readSlice()
}

func ReadSliceFile(fname string) (*Slice, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadSlice(f)
}

// Reads successive data frames in dump format.
type reader struct {
	header
	in  io.Reader
	crc hash.Hash64
	err error
}

func newReader(in io.Reader) *reader {
	r := new(reader)
	r.in = in
	r.crc = crc64.New(table)
	return r
}

func (r *reader) readSlice() (slice *Slice, err error) {
	r.err = nil // clear previous error, if any
	r.Magic = r.readString()
	if r.err != nil {
		return nil, r.err
	}
	if r.Magic != MAGIC {
		r.err = fmt.Errorf("dump: bad magic number:%v", r.Magic)
		return nil, r.err
	}
	r.Components = r.readInt()
	for i := range r.MeshSize {
		r.MeshSize[i] = r.readInt()
	}
	for i := range r.MeshStep {
		r.MeshStep[i] = r.readFloat64()
	}
	r.MeshUnit = r.readString()
	r.Time = r.readFloat64()
	r.TimeUnit = r.readString()
	r.DataLabel = r.readString()
	r.DataUnit = r.readString()
	r.Precission = r.readUint64()

	for i := 0; i < padding; i++ {
		_ = r.readUint64()
	}

	if r.err != nil {
		return
	}

	slice, err = r.readData()

	// Check CRC
	var mycrc uint64 // checksum by this reader
	if r.crc != nil {
		mycrc = r.crc.Sum64()
	}
	storedcrc := r.readUint64() // checksum from data stream. 0 means not set

	if r.crc != nil {
		r.crc.Reset() // reset for next frame
	}

	if r.crc != nil && storedcrc != 0 &&
		mycrc != storedcrc &&
		r.err == nil {
		r.err = fmt.Errorf("dump CRC error: expected %16x, got %16x", storedcrc, mycrc)
	}
	return
}

func (r *reader) readInt() int {
	x := r.readUint64()
	if uint64(int(x)) != x {
		r.err = fmt.Errorf("value overflows int: %v", x)
	}
	return int(x)
}

// read until the buffer is full
func (r *reader) read(buf []byte) {
	_, err := io.ReadFull(r.in, buf[:])
	if err != nil {
		r.err = err
	}
	if r.crc != nil {
		r.crc.Write(buf)
	}
}

// read a maximum 8-byte string
func (r *reader) readString() string {
	var buf [8]byte
	r.read(buf[:])
	// trim trailing NULs.
	i := 0
	for i = 0; i < len(buf); i++ {
		if buf[i] == 0 {
			break
		}
	}
	return string(buf[:i])
}

func (r *reader) readFloat64() float64 {
	return math.Float64frombits(r.readUint64())
}

func (r *reader) readUint64() uint64 {
	var buf [8]byte
	r.read(buf[:])
	return *((*uint64)(unsafe.Pointer(&buf[0])))
}

// read the data array,
// enlarging the previous one if needed.
func (r *reader) readData() (*Slice, error) {
	s := r.MeshSize
	c := r.MeshStep
	mesh := NewMesh(s[0], s[1], s[2], c[0], c[1], c[2])
	length := mesh.NCell()
	slice := NewSlice(r.Components, mesh)
	host := slice.Host()
	for _, data := range host {
		buf := (*(*[1<<31 - 1]byte)(unsafe.Pointer(&data[0])))[0 : SIZEOF_FLOAT32*length]
		r.read(buf)
	}
	if r.err == nil {
		return slice, nil
	}
	return nil, r.err
}
