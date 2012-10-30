package core

import "fmt"

type chandata struct {
	*Info
	slice Slice
	array [][][]float32
	list  []float32
}

func makedata(tag, unit string, m *Mesh, blocks ...int) chandata {
	size := m.Size()
	var c chandata
	c.Info = NewInfo(tag, unit, m, blocks...)
	c.array = MakeFloats(size)
	c.list = Contiguous(c.array)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chandata) UnsafeData() []float32      { return d.list }
func (d *chandata) UnsafeArray() [][][]float32 { return d.array }
func (d *chandata) Size() [3]int               { return SizeOf(d.array) }
func (d *chandata) NComp() int                 { return len(d.list) }

func (q Chan1) String() string {
	unit := q.unit
	if unit != "" {
		unit = " [" + unit + "]"
	}
	return fmt.Sprint(q.tag, unit, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")
}
