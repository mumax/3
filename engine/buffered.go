package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

// Output Handle for a quantity that is stored on the GPU.
type buffered struct {
	data.Synced
	autosave
}

func newBuffered(nComp int, name string) *buffered {
	b := new(buffered)
	b.Synced = *data.NewSynced(cuda.NewSlice(nComp, mesh))
	b.name = name
	return b
}

// notify the handle that it may need to be saved
func (b *buffered) touch(goodstep bool) {
	if goodstep && b.needSave() {
		GoSave(b.fname(), b.Read(), Time, func() { b.ReadDone() })
		b.saved()
	}
}

// Memset with synchronization.
func (b *buffered) memset(val ...float32) {
	s := b.Write()
	cuda.Memset(s, val...)
	b.WriteDone()
}

// Normalize with synchronization.
func (b *buffered) normalize() {
	s := b.Write()
	cuda.Normalize(s)
	b.WriteDone()
}
