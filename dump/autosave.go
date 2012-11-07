package dump

import (
	"code.google.com/p/nimble-cube/nimble"
)

type Autosaver struct {
	out   *Writer
	data  nimble.RChan3
	every int
}

func NewAutosaver(fname string, data nimble.RChan3, every int) *Autosaver {
	r := new(Autosaver)
	r.out = NewWriter(nimble.OpenFile(nimble.OD+fname), CRC_ENABLED)
	nimble.Assert(data.NComp() == 3)
	r.out.Components = 3 // TODO !!
	r.out.MeshSize = data.Mesh().Size()
	r.data = data
	r.every = every
	return r
}

func (r *Autosaver) Run() {
	N := nimble.Prod(r.data.Mesh().Size())

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO
		if i%r.every == 0 {
			i = 0
			nimble.Debug("dump")
			r.out.WriteHeader()
			r.out.WriteData(output[0].Host())
			r.out.WriteData(output[1].Host())
			r.out.WriteData(output[2].Host())
			r.out.WriteHash()
		}
		r.data.ReadDone()
	}
}
