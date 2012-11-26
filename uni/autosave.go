package uni

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"code.google.com/p/nimble-cube/nimble"
)

type Autosaver struct {
	out   *dump.Writer
	data  nimble.RChanN
	every int
	hostBuf
	Dev Device
}

func Autosave(data_ nimble.Chan, every int, dev Device) {
	fname := data_.ChanN().Tag() + ".dump"
	r := new(Autosaver)
	r.out = dump.NewWriter(core.OpenFile(core.OD+fname), dump.CRC_ENABLED)
	data := data_.ChanN().NewReader()
	r.out.Components = data.NComp()
	r.out.MeshSize = data.Mesh().Size()
	r.data = data
	r.every = every
	r.Dev = dev
	nimble.Stack(r)
}

func (r *Autosaver) Run() {
	N := r.data.Mesh().NCell()

	if !r.data.MemType().CPUAccess(){
		core.Assert(r.Dev != nil)
		r.Dev.InitThread()
	}

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO: could read comp by comp...
		if i%r.every == 0 {
			i = 0
			core.Debug("dump")
			r.out.WriteHeader()
			for c := 0; c < r.data.NComp(); c++ {
				r.out.WriteData(r.gethost(output[c]))
			}
			r.out.WriteHash()
		}
		r.data.ReadDone()
	}
}

//func (r*Autosaver) gethost(data Slice) []float32{
//if data.MemType().CPUAccess()
//}

type hostBuf []float32

func (r *hostBuf) gethost(data nimble.Slice) []float32 {
	if data.CPUAccess() {
		return data.Host()
	} // else
	if *r == nil {
		core.Debug("alloc buffer")
		*r = make([]float32, data.Len())
	}
	data.Device().CopyDtoH(*r)
	return *r
}
