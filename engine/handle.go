package engine

import (
	"code.google.com/p/mx3/data"
)

type Handle interface {
}

type AddFn func(dst *data.Slice)

type Adder struct {
	addFn AddFn // calculates quantity and add result to dst
	autosave
}

func NewAdder(name string, f AddFn) *Adder {
	a := new(Adder)
	a.addFn = f
	a.name = name
	return a
}
