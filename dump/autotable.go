package dump

import (
	"nimble-cube/core"
)

type Autotable struct {
	out   TableWriter
	data  core.RChan3
	every int
}

/* TODO:
idea to get the time:
rwmutex can give you the step (from absA and prod(size), e.g)
solver could update small map with step: time
*/

func NewAutotable(fname string, data core.RChan3, every int) *Autotable {
	r := new(Autotable)
	tags := []string{data[0].Tag(), data[1].Tag(), data[2].Tag()}
	units := []string{"?", "?", "?"} // TODO
	r.out = NewTableWriter(core.OpenFile(core.OD+fname), tags, units)
	r.data = data
	r.every = every
	return r
}

func (r *Autotable) Run() {
	core.Log("running auto table")
	N := core.Prod(r.data.Size())

	for i := 0; ; i++ {
		output := r.data.ReadNext(N) // TODO
		if i%r.every == 0 {
			i = 0
			for c := range output {
				sum := 0.
				for j := range output[c] {
					sum += float64(output[c][j])
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
