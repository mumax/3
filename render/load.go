package render

import (
	"code.google.com/p/mx3/core"
	"code.google.com/p/mx3/dump"
	"os"
)

var Frame *dump.Frame

func Load(fname string) {
	core.Log("loading", fname)
	f, err := os.Open(fname)
	core.Fatal(err)
	defer f.Close()
	r := dump.NewReader(f, dump.CRC_ENABLED)
	core.Fatal(r.Read())
	core.Log("loaded", fname)
	Frame = &(r.Frame)
	Crop2 = Frame.MeshSize
}
