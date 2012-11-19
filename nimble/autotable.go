package nimble

import (
	"code.google.com/p/nimble-cube/core"
	"code.google.com/p/nimble-cube/dump"
)

type Autotabler struct {
	out   dump.TableWriter
	data  RChanN
	every int
}

func Autotable(data_ Chan, every int) {
	r := new(Autotabler)
	data := data_.ChanN().NewReader()
	tags := make([]string, data.NComp())
	units := make([]string, data.NComp())
	for i := range tags {
		tags[i] = data.Comp(i).Tag()
		units[i] = data.Comp(i).Unit()
	}
	fname := data.Tag() + ".table"
	r.out = dump.NewTableWriter(core.OpenFile(core.OD+fname), tags, units)
	r.data = data
	r.every = every
	Stack(r)
}

func (r *Autotabler) Run() {
	core.Log("running auto table")
	N := core.Prod(r.data.Mesh().Size())

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO
		if i%r.every == 0 {
			i = 0
			for c := range output {
				sum := 0.
				list := output[c].Host()
				for j := range list {
					sum += float64(list[j])
				}
				sum /= float64(N)
				r.out.Data[c] = float32(sum)
			}
		}
		r.data.ReadDone()
		r.out.WriteData()
		r.out.Flush()
	}
}
