// legacy dump data format.
package dump

import (
	"fmt"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"os"
	"unsafe"
)

func Read(in io.Reader) (*data.Slice, data.Meta, error) {
	r := newReader(in)
	return r.readSlice()
}

func ReadFile(fname string) (*data.Slice, data.Meta, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, data.Meta{}, err
	}
	defer f.Close()
	return Read(f)
}

func MustReadFile(fname string) (*data.Slice, data.Meta) {
	s, t, err := ReadFile(fname)
	util.FatalErr(err)
	return s, t
}

// Reads successive data frames in dump format.
type reader struct {
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

func (r *reader) readSlice() (s *data.Slice, info data.Meta, err error) {
	r.err = nil // clear previous error, if any
	magic := r.readString()
	if r.err != nil {
		return nil, data.Meta{}, r.err
	}
	if magic != MAGIC {
		r.err = fmt.Errorf("dump: bad magic number:%v", magic)
		return nil, data.Meta{}, r.err
	}
	nComp := r.readInt()
	size := [3]int{}
	size[2] = r.readInt() // backwards compatible coordinates!
	size[1] = r.readInt()
	size[0] = r.readInt()
	cell := [3]float64{}
	cell[2] = r.readFloat64()
	cell[1] = r.readFloat64()
	cell[0] = r.readFloat64()
	info.CellSize = cell

	info.MeshUnit = r.readString()
	info.Time = r.readFloat64()
	_ = r.readString() // time unit

	s = data.NewSlice(nComp, size)

	info.Name = r.readString()
	info.Unit = r.readString()
	precission := r.readUint64()
	util.AssertMsg(precission == 4, "only single precission supported")

	if r.err != nil {
		return
	}

	host := s.Tensors()
	ncomp := s.NComp()
	for c := 0; c < ncomp; c++ {
		for iz := 0; iz < size[2]; iz++ {
			for iy := 0; iy < size[1]; iy++ {
				for ix := 0; ix < size[0]; ix++ {
					host[c][iz][iy][ix] = r.readFloat32()
				}
			}
		}
	}

	// Check CRC
	var mycrc uint64 // checksum by this reader
	if r.crc != nil {
		mycrc = r.crc.Sum64()
	}
	storedcrc := r.readUint64() // checksum from data stream. 0 means not set
	if r.err != nil {
		return nil, data.Meta{}, r.err
	}
	if r.crc != nil {
		r.crc.Reset() // reset for next frame
	}
	if r.crc != nil && storedcrc != 0 && mycrc != storedcrc {
		r.err = fmt.Errorf("dump CRC error: expected %16x, got %16x", storedcrc, mycrc)
		return nil, data.Meta{}, r.err
	}

	return s, info, nil
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

func (r *reader) readFloat32() float32 {
	var buf [4]byte
	r.read(buf[:])
	return *((*float32)(unsafe.Pointer(&buf[0])))
}
