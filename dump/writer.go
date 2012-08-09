package dump

import (
	"hash"
	"hash/crc64"
	"io"
	"encoding/binary"
	"math"
)

var table = crc64.MakeTable(crc64.ISO)

type Writer struct {
	Header
	out io.Writer
	crc hash.Hash
}

func NewWriter(out io.Writer) *Writer {
	w := new(Writer)
	w.crc = crc64.New(table)
	w.out = io.MultiWriter(w.crc, out)
	return w
}

func (w *Writer) WriteHeader() {
	w.writeString(MAGIC)
	w.writeString(w.TimeLabel)
	w.writeFloat64(w.Time)
	w.writeString(w.SpaceLabel)
	for _, c:=range w.CellSize{
		w.writeFloat64(c)
	}
	w.writeInt(w.Rank)
	for _,s:=range w.Size{
	w.writeInt(s)
	}
	w.writeInt(FLOAT32)
	
}

func(w*Writer)count(n int, err error){

}

func(w*Writer)writeFloat64(x float64){
	w.writeUInt64(math.Float64bits(x)
}

func(w*Writer)writeString(x string){
	w.count(w.out.Write(	
}
func(w*Writer)writeUInt64(x uint64){
	var buf [8]byte
	binary.PutUvarint(buf[:], x)
	w.count(w.out.Write(buf[:]))
}

func (w *Writer) WriteData(data []float32) error {
	return nil
}

func (w *Writer) WriteHash() error {
	return nil
}
