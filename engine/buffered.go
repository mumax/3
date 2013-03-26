package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"sync"
)

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
