package dump

import (
	"nimble-cube/nimble"
)

type Autotable struct {
	out   TableWriter
	data  nimble.RChan3
	every int
}

/* TODO:
idea to get the time:
rwmutex can give you the step (from absA and prod(size), e.g)
solver could update small map with step: time
*/

func NewAutotable(fname string, data nimble.RChan3, every int) *Autotable {
	r := new(Autotable)
	tags := []string{data[0].Tag(), data[1].Tag(), data[2].Tag()}
	units := []string{"?", "?", "?"} // TODO
	r.out = NewTableWriter(nimble.OpenFile(nimble.OD+fname), tags, units)
	r.data = data
	r.every = every
	return r
}

func (r *Autotable) Run() {
	nimble.Log("running auto table")
	N := nimble.Prod(r.data.Mesh().Size())

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
