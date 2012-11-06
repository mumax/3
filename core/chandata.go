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
	size := m.Size()
	var c chandata
	c.Info = NewInfo(tag, unit, m, blocks...)
	c.slice.array = MakeFloats(size)
	c.slice.list = Contiguous(c.slice.array)
	N := len(c.slice.list)
	c.slice.gpu.UnsafeSet(unsafe.Pointer(&c.slice.list[0]), N, N)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chandata) UnsafeData() []float32      { return d.slice.list }
func (d *chandata) UnsafeArray() [][][]float32 { return d.slice.array }
func (d *chandata) Size() [3]int               { return SizeOf(d.slice.array) }
func (d *chandata) NComp() int                 { return len(d.slice.list) }

func (q Chan1) String() string {
	unit := q.unit
	if unit != "" {
		unit = " [" + unit + "]"
	}
	return fmt.Sprint(q.tag, unit, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")
}
