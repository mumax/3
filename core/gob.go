package core

// Save/Load gob files

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"reflect"
)

type Tensor struct {
	Size [4]int
}

func Save(v interface{}, file string) {
	Debug("saving to", file)
	out_, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	Fatal(err)
	out := bufio.NewWriter(out_)
	enc := json.NewEncoder(out)
	PanicErr(enc.Encode(HeaderOf(v)))
	PanicErr(binary.Write(out, binary.BigEndian, v))
	PanicErr(out.Flush())
	PanicErr(out_.Close())
}

func HeaderOf(x interface{}) Tensor {
	switch v := x.(type) {
	case [3][][][]float32:
		return Tensor{[4]int{len(v), len(v[0]), len(v[0][0]), len(v[0][0][0])}}
	default:
		Panic("Unhandled type:", reflect.TypeOf(v))
	}
	var t Tensor
	return t
}

func Read(in io.Reader) interface{} {
	//	dec := gob.NewDecoder(in)
	//	var v interface{}
	//	PanicErr(dec.Decode(&v))
	return nil
}
