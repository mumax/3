package main

import (
	"code.google.com/p/mx3/data"
	"encoding/json"
	"io"
)

func dumpJSON(out io.Writer, f *data.Slice) {
	w := json.NewEncoder(out)
	w.Encode(f.Tensors())
}
