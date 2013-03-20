package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
)

// TODO: what if we want to save energies etc?
type Quant struct {
	addTo func(dst *data.Slice) // adds quantity to dst
	autosave
}

func NewQuant(name string, adder func(dst *data.Slice)) Quant {
	return Quant{addTo: adder, autosave: autosave{name: name}}
}

func (q *Quant) AddTo(dst *data.Slice) {
	if Solver.GoodStep && q.needSave() {
		buffer := OutputBuffer(dst.NComp())
		q.addTo(buffer)
		cuda.Madd2(dst, dst, buffer, 1, 1)
		go SaveAndRecycle(buffer, q.fname(), Time)
		q.autosave.count++ // !
	} else {
		q.addTo(dst)
	}
}

type autosave struct {
	period float64 // How often to save
	start  float64 // Starting point
	count  int     // Number of times it has been saved
	name   string
}

func (a *autosave) Autosave(period float64) {
	a.period = period
	a.start = Time
	a.count = 0
}

func (a *autosave) needSave() bool {
	if a.period == 0 {
		return false
	}
	t := Time - a.start
	return t-float64(a.count)*a.period >= a.period
}

func (a *autosave) fname() string {
	return fmt.Sprintf("%s%s%06d.dump", OD, a.name, a.count)
}
