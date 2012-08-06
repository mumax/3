package core

// Save/Load gob files

import (
	"bufio"
	"encoding/gob"
	"io"
	"os"
)

func Save(v interface{}, file string) {
	Debug("saving to", file)
	out_, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	Fatal(err)
	out := bufio.NewWriter(out_)
	enc := gob.NewEncoder(out)
	PanicErr(enc.Encode(&v))
	PanicErr(out.Flush())
	PanicErr(out_.Close())
}

func Read(in io.Reader) interface{} {
	dec := gob.NewDecoder(in)
	var v interface{}
	PanicErr(dec.Decode(&v))
	return v
}

// Register interface types for gob transmission
func init(){
	Debug("Initializing gob")
	var a [3][][][]float32
	gob.Register(a)
}
