package main

import (
	"encoding/json"
	"github.com/mumax/3/data"
	"io"
)

func dumpJSON(out io.Writer, f *data.Slice) {
	w := json.NewEncoder(out)
	w.Encode(f.Tensors())
}
