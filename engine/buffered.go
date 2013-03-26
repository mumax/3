package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
	"fmt"
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

func (b *Buffered) Write() *data.Slice {
	b.lock.Lock()
	return b.buffer
}

func (b *Buffered) WriteDone() {
	b.lock.Unlock()
}

func (b *Buffered) Memset(val ...float32) {
	s := b.Write()
	cuda.Memset(s, val...)
	b.WriteDone()
}

// autosaves if needed
func (b *Buffered) Touch() {
	if b.needSave() {
		fmt.Println("save", b.name)
		b.saved() // count++ hazard
	}
}
