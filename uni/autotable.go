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
	time  <-chan nimble.Time
	hostBuf
}

func Autotable(data_ nimble.Chan, every int, dev Device) {
	r := new(Autotabler)
	data := data_.ChanN().NewReader()
	tags := make([]string, data.NComp()+1)
	units := make([]string, data.NComp()+1)
	tags[0], units[0] = "t", "s"
	for i := 0; i < data.NComp(); i++ {
		cmp := data.Comp(i) // gccgo hack: gccgo 4.7.2 wants a pointer here
		tags[i+1] = (&cmp).Tag()
		units[i+1] = (&cmp).Unit()
	}
	fname := data.Tag() + ".table"
	r.out = dump.NewTableWriter(core.OpenFile(core.OD+fname), tags, units)
	r.data = data
	r.every = every
	r.Dev = dev
	r.time = nimble.Clock.NewReader()
	nimble.Stack(r)
}

func (r *Autotabler) Run() {
	core.Log("running auto table")
	N := core.Prod(r.data.Mesh().Size())

	if !r.data.MemType().CPUAccess() {
		core.Assert(r.Dev != nil)
		r.Dev.InitThread()
	}

	i:=0
	for {
		output := r.data.ReadNext(N) // TODO
		time := <-r.time
		if time.Stage && i%r.every == 0 {
			i = 0
			r.out.Data[0] = float32(time.Time)
			for c := range output {
				sum := 0.
				list := r.gethost(output[c])
				for j := range list {
					sum += float64(list[j])
				}
				sum /= float64(N)
				r.out.Data[c+1] = float32(sum)
			}
			r.out.Flush()
			r.out.WriteData()
		}
		r.data.ReadDone()
		if time.Stage{i++}
	}
}
