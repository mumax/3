package core

import (
	"fmt"
	"unsafe"
)

type chandata struct {
	*Info
	slice Slice
}

func makedata(tag, unit string, m *Mesh, blocks ...int) chandata {
	var c chandata
	c.Info = NewInfo(tag, unit, m, blocks...)
	N := m.NCell()                    // TODO: block len
	c.slice.list = make([]float32, N) //Contiguous(c.slice.array)
	c.slice.gpu.UnsafeSet(unsafe.Pointer(&c.slice.list[0]), N, N)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chandata) UnsafeData() []float32 { return d.slice.list }

func (d *chandata) NComp() int { return len(d.slice.list) }

func (q Chan1) String() string {
	unit := q.unit
	if unit != "" {
		unit = " [" + unit + "]"
	}
	return fmt.Sprint(q.tag, unit, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")
}

func (c chandata) UnsafeArray() [][][]float32 {
	return Reshape(c.slice.Host(), c.Size())
}
