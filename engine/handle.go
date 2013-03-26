package engine

import (
	"code.google.com/p/mx3/data"
	"code.google.com/p/mx3/cuda"
)

type Handle interface {
}

type Buffered struct {
	buffer *data.Slice
	autosave
}

func NewBuffered(nComp int, name string)*Buffered{
	b := new(Buffered)
	b.buffer = cuda.NewSlice(nComp, mesh)
	b.name = name
	return b
}


type AddFn func(dst *data.Slice)

type Adder struct {
	addFn  AddFn // calculates quantity and add result to dst
}

func NewAdder(f AddFn)*Adder{
	a := new(Adder)
	a.addFn = f
	return a	
}
