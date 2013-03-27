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

// autosaves if needed
func (b *Buffered) Touch() {
	if b.needSave() {
		fmt.Println("save", b.name)
		b.saved() // count++ hazard
	}
}
