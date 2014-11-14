package main

import (
	"encoding/json"
	"io"

	"github.com/mumax/3/data"
)

func dumpJSON(out io.Writer, f *data.Slice) {
	w := json.NewEncoder(out)
	w.Encode(f.Tensors())
}
