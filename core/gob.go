package core

// Save/Load gob files

import (
	"bufio"
	"encoding/gob"
	"os"
)

func Save(v interface{}, file string) {
	out_, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	PanicErr(err)
	out := bufio.NewWriter(out_)
	enc := gob.NewEncoder(out)
	PanicErr(enc.Encode(v))
	PanicErr(out.Flush())
	PanicErr(out_.Close())
}
