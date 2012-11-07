package dump

import (
	"bufio"
	"io"
	"math"
	"nimble-cube/nimble"
	"unsafe"
)

// Magic number
const TABLE_MAGIC = "table002"

type TableWriter struct {
	out  *bufio.Writer
	Data []float32
}

func NewTableWriter(out io.Writer, tags, units []string) TableWriter {
	if len(tags) != len(units) {
		nimble.Panic("table: len(tags) != len(units)")
	}

	bufout, ok := out.(*bufio.Writer)
	if !ok {
		bufout = bufio.NewWriter(out)
	}

	var t TableWriter
	t.out = bufout
	t.Data = make([]float32, len(tags))

	t.writeString(TABLE_MAGIC)
	t.writeInt(len(tags))
	for i := range tags {
		t.writeString(tags[i])
		t.writeString(units[i])
	}
	precission := FLOAT32
	t.writeInt(precission)
	t.out.Flush()
	return t
}

func (w *TableWriter) WriteData() {
	list := w.Data
	w.check(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)]))
}

func (w *TableWriter) Flush() {
	nimble.PanicErr(w.out.Flush())
}
func (w *TableWriter) writeFloat64(x float64) {
	w.writeUInt64(math.Float64bits(x))
}

func (w *TableWriter) writeFloat32(x float32) {
	w.writeUInt32(math.Float32bits(x))
}

func (w *TableWriter) writeString(x string) {
	var buf [8]byte
	copy(buf[:], x)
	w.check(w.out.Write(buf[:]))
}

func (w *TableWriter) writeUInt64(x uint64) {
	w.check(w.out.Write((*(*[8]byte)(unsafe.Pointer(&x)))[:8]))
}

func (w *TableWriter) writeUInt32(x uint32) {
	w.check(w.out.Write((*(*[4]byte)(unsafe.Pointer(&x)))[:4]))
}

func (w *TableWriter) writeInt(x int) {
	w.writeUInt64(uint64(x))
}
func (w *TableWriter) check(n int, err error) {
	if err != nil {
		nimble.Panic(err)
	}
}
