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
	magic := r.readString()
	if magic != MAGIC {
		r.Err = fmt.Errorf("dump: bad magic number:%v", magic)
		return
	}
	r.TimeLabel = r.readString()
	r.Time = r.readFloat64()
	r.SpaceLabel = r.readString()
	for i := range r.CellSize {
		r.CellSize[i] = r.readFloat64()
	}
	r.Rank = int(r.readUint64())
	r.Size = make([]int, r.Rank)
	for i := 0; i < r.Rank; i++ {
		r.Size[i] = int(r.readUint64())
	}
	r.Precission = r.readUint64()

	r.readData()

	mycrc := r.crc.Sum64()
	r.CRC = r.readUint64()
	if mycrc != r.CRC && r.Err == nil {
		r.Err = fmt.Errorf("dump CRC error: expected %x, got %x", r.CRC, mycrc)
	}
	r.crc.Reset()
}

//w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)]))

func (r *Reader) read(buf []byte) {
	n, err := io.ReadFull(r.in, buf[:])
	r.Bytes += int64(n)
	if err != nil {
		r.Err = err
	}
	n, err = r.crc.Write(buf)
	if err != nil {
		panic(err)
	}
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

func (r *Reader) readFloat64() float64 {
	return math.Float64frombits(r.readUint64())
}

func (r *Reader) readUint64() uint64 {
	var buf [8]byte
	r.read(buf[:])
	return *((*uint64)(unsafe.Pointer(&buf[0])))
}

func (r *Reader) readData() {
	N := 1
	for _, s := range r.Size {
		N *= s
	}
	if cap(r.Data) < N {
		r.Data = make([]float32, N)
	}
	if len(r.Data) < N {
		r.Data = r.Data[:N]
	}
	buf := (*(*[1<<31 - 1]byte)(unsafe.Pointer(&r.Data[0])))[0 : 4*len(r.Data)]
	r.read(buf)
}

func (r *Reader) Fprint(out io.Writer) {
	if r.Err != nil {
		fmt.Fprintln(out, r.Err)
		return
	}
	fmt.Fprintf(out, "%#v\n", r.Header)
	fmt.Fprintf(out, "Data%v\n", r.Data)
	fmt.Fprintf(out, "ISO CRC64:%x\n", r.CRC)
}
