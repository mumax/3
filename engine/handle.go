package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"sync"
)

type Handle interface {
}

type Buffered struct {
	buffer *data.Slice
	lock   sync.RWMutex
	autosave
}

func NewBuffered(nComp int, name string) *Buffered {
	b := new(Buffered)
	b.buffer = cuda.NewSlice(nComp, mesh)
	b.name = name
	return b
}

func (b *Buffered) Read() *data.Slice {
	b.lock.RLock()
	return b.buffer
}

func (b *Buffered) ReadDone() {
	b.lock.RUnlock()
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
