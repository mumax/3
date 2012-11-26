package uni

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
	"code.google.com/p/nimble-cube/nimble"
)

type Autotabler struct {
	out   dump.TableWriter
	data  nimble.RChanN
	every int
	Dev   Device
	hostBuf
}

func Autotable(data_ nimble.Chan, every int, dev Device) {
	r := new(Autotabler)
	data := data_.ChanN().NewReader()
	tags := make([]string, data.NComp())
	units := make([]string, data.NComp())
	for i := range tags {
		cmp := data.Comp(i) // gccgo hack: gccgo 4.7.2 wants a pointer here
		tags[i] = (&cmp).Tag()
		units[i] = (&cmp).Unit()
	}
	fname := data.Tag() + ".table"
	r.out = dump.NewTableWriter(core.OpenFile(core.OD+fname), tags, units)
	r.data = data
	r.every = every
	r.Dev = dev
	nimble.Stack(r)
}

func (r *Autotabler) Run() {
	core.Log("running auto table")
	N := core.Prod(r.data.Mesh().Size())

	if !r.data.MemType().CPUAccess() {
		core.Assert(r.Dev != nil)
		r.Dev.InitThread()
	}

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO
		if i%r.every == 0 {
			i = 0
			for c := range output {
				sum := 0.
				list := r.gethost(output[c])
				for j := range list {
					sum += float64(list[j])
				}
				sum /= float64(N)
				r.out.Data[c] = float32(sum)
			}
			r.out.Flush()
			r.out.WriteData()
		}
		r.data.ReadDone()
	}
}
