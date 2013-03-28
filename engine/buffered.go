package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

type Buffered struct {
	data.Synced
	autosave
}

func NewBuffered(nComp int, name string) *Buffered {
	b := new(Buffered)
	b.Synced = *data.NewSynced(cuda.NewSlice(nComp, mesh))
	b.name = name
	return b
}

// Autosaves if needed.
func (b *Buffered) Touch() {
	if b.needSave() {
		GoSave(b.fname(), b.Read(), Time, func() { b.ReadDone() })
		b.saved() // count++ hazard?
	}
}

// Memset with synchronization.
func (b *Buffered) Memset(val ...float32) {
	s := b.Write()
	cuda.Memset(s, val...)
	b.WriteDone()
}

// Normalize with synchronization.
func (b *Buffered) Normalize() {
	s := b.Write()
	cuda.Normalize(s)
	b.WriteDone()
}
