package dump

import(
	"nimble-cube/core"
	"io"
	"math"
	"unsafe"
	"bufio"
)

type Table struct{
	out *bufio.Writer
}

func NewTable(out io.Writer, tags, units []string) Table{
	bufout, ok := out.(*bufio.Writer)
	if !ok{
		bufout = bufio.NewWriter(out)	
	}

	var t Table
	t.out = bufout

	t

	return t
}





func (w *Table) writeFloat64(x float64) {
	w.writeUInt64(math.Float64bits(x))
}

func (w *Table) writeFloat32(x float32) {
	w.writeUInt32(math.Float32bits(x))
}

func (w *Table) writeString(x string) {
	var buf [8]byte
	copy(buf[:], x)
	w.check(w.out.Write(buf[:]))
}

func (w *Table) writeUInt64(x uint64) {
	w.check(w.out.Write((*(*[8]byte)(unsafe.Pointer(&x)))[:8]))
}

func (w *Table) writeUInt32(x uint32) {
	w.check(w.out.Write((*(*[4]byte)(unsafe.Pointer(&x)))[:4]))
}

func(w*Table)check(n int, err error){
	if err != nil{core.Panic(err)}
}
