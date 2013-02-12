package io

import (
	"code.google.com/p/mx3/mx"
	"fmt"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"unsafe"
)

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

// Reads one frame and stores it in r.Frame.
func (r *reader) readFrame() error {
	r.err = nil // clear previous error, if any
	r.Magic = r.readString()
	if r.err != nil {
		return r.err
	}
	if r.Magic != MAGIC {
		r.err = fmt.Errorf("dump: bad magic number:%v", r.Magic)
		return r.err
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
	if r.err != nil {
		return r.err
	}

	r.readData()

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
	return r.err
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
	n, err := io.ReadFull(r.in, buf[:])
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
func (r *reader) readData() {
	s := r.MeshSize
	c := r.MeshStep
	mesh := mx.NewMesh(s[0], s[1], s[2], c[0], c[1], c[2])
	slice := mx.NewCPUSlice(r.Components, mesh)
	for c := 0; c < nComp; c++ {
		buf := (*(*[1<<31 - 1]byte)(unsafe.Pointer(&r.Data[0])))[0 : 4*len(r.Data)]
	}
	r.read(buf)
}
