package dump

import (
	"nimble-cube/core"
)

type Autosaver struct {
	out   *Writer
	data  core.RChan3
	every int
}

func NewAutosaver(fname string, data core.RChan3, every int) *Autosaver {
	r := new(Autosaver)
	r.out = NewWriter(core.OpenFile(fname), CRC_ENABLED)
	r.out.Components = 3 // TODO !!
	r.out.MeshSize = data.Size()
	r.data = data
	r.every = every
	return r
}

func (r *Autosaver) Run() {
	N := core.Prod(r.data.Size())

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO
		if i%r.every == 0 {
			i = 0
			core.Debug("dump")
			r.out.WriteHeader()
			r.out.WriteData(output[0])
			r.out.WriteData(output[1])
			r.out.WriteData(output[2])
			r.out.WriteHash()
		}
		r.data.ReadDone()
	}
}
