package nimble

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
)

type Autosaver struct {
	out   *dump.Writer
	data  RChanN
	every int
}

func RunAutosaver(fname string, data_ Chan, every int) *Autosaver {
	r := new(Autosaver)
	r.out = dump.NewWriter(core.OpenFile(core.OD+fname), dump.CRC_ENABLED)
	data := data_.ChanN().NewReader()
	r.out.Components = data.NComp()
	r.out.MeshSize = data.Mesh().Size()
	r.data = data
	r.every = every
	Stack(r)
	return r
}

func (r *Autosaver) Run() {
	N := r.data.Mesh().NCell()

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO: could read comp by comp...
		if i%r.every == 0 {
			i = 0
			core.Debug("dump")
			r.out.WriteHeader()
			for c := 0; c < r.data.NComp(); c++ {
				r.out.WriteData(output[c].Host())
			}
			r.out.WriteHash()
		}
		r.data.ReadDone()
	}
}
