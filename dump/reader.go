package dump

import (
	"fmt"
	"hash"
	"hash/crc64"
	"io"
	"math"
	"unsafe"
)

type Reader struct {
	Frame
	Bytes int64 // Total number of bytes read.
	Err   error // Stores the latest I/O error, if any.
	in    io.Reader
	crc   hash.Hash64
}

func NewReader(in io.Reader) *Reader {
	r := new(Reader)
	r.in = in
	r.crc = crc64.New(table)
	return r
}

func (r *Reader) Read() {
	//magic := r.readString()
	r.TimeLabel = r.readString()
	//r.writeFloat64(w.Time)
	//r.writeString(w.SpaceLabel)
	//for _, c := range w.CellSize {
	//	w.writeFloat64(c)
	//}
	//w.writeUInt64(uint64(w.Rank))
	//for _, s := range w.Size {
	//	w.writeUInt64(uint64(s))
	//}
	//w.writeUInt64(FLOAT32)
}

//w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)]))

func (r *Reader) read(buf []byte) {
	n, err := io.ReadFull(r.in, buf[:])
	r.Bytes += int64(n)
	if err != nil {
		r.Err = err
	}
	r.crc.Write(buf)
}

func (r *Reader) readString() string {
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

func (r *Reader) ReadFloat64() float64 {
	return math.Float64frombits(r.readUint64())
}

func (r *Reader) readUint64() uint64 {
	var buf [8]byte
	r.read(buf[:])
	return *((*uint64)(unsafe.Pointer(&buf[0])))
}

func (r *Reader) Fprint(out io.Writer) {
	fmt.Fprintf(out, "%#v\n", r.Header)
	fmt.Fprintf(out, "dump.Data%v\n", r.Data)
	fmt.Fprintf(out, "dump.CRC%v\n", r.CRC)
}
