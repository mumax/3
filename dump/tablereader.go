package dump

import (
	"bufio"
	"code.google.com/p/mx3/core"
	"io"
	"math"
	"unsafe"
)

type TableReader struct {
	in        *bufio.Reader
	Data      []float32
	Tags      []string
	Units     []string
	dataBytes []byte
}

func NewTableReader(in io.Reader) TableReader {

	bufin, ok := in.(*bufio.Reader)
	if !ok {
		bufin = bufio.NewReader(in)
	}

	var t TableReader
	t.in = bufin
	magic := t.readString()
	if magic != TABLE_MAGIC {
		core.Panic("bad magic number:", magic)
	}

	n := t.readInt()
	t.Data = make([]float32, n)
	t.Tags = make([]string, n)
	t.Units = make([]string, n)

	for i := range t.Tags {
		t.Tags[i] = t.readString()
		t.Units[i] = t.readString()
	}

	precission := t.readInt()
	if precission != FLOAT32 {
		core.Panic("only 32 bit is supported")
	}
	// t.Data exposed as raw bytes
	t.dataBytes = (*(*[1<<31 - 1]byte)(unsafe.Pointer(&t.Data[0])))[0 : 4*len(t.Data)]

	return t
}

func (r *TableReader) ReadLine() error {
	_, err := io.ReadFull(r.in, r.dataBytes)
	if err != nil {
		return err
	}
	return nil
}

func (r *TableReader) readInt() int {
	x := r.readUint64()
	if uint64(int(x)) != x {
		core.Panic("value overflows int:", x)
	}
	return int(x)
}

// read until the buffer is full
func (r *TableReader) read(buf []byte) {
	_, err := io.ReadFull(r.in, buf[:])
	core.PanicErr(err)
}

// read a maximum 8-byte string
func (r *TableReader) readString() string {
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

func (r *TableReader) readFloat64() float64 {
	return math.Float64frombits(r.readUint64())
}

func (r *TableReader) readUint64() uint64 {
	var buf [8]byte
	r.read(buf[:])
	return *((*uint64)(unsafe.Pointer(&buf[0])))
}
