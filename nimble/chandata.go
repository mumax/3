package nimble

import (
//"fmt"
)

//type chandata struct {
//	*Info
//	slice Slice
//}
//
//func makedata(tag, unit string, m *Mesh, memtype MemType, blocks ...int) chandata {
//	var c chandata
//	c.Info = NewInfo(tag, unit, m, blocks...)
//	N := m.NCell() // TODO: block len
//
//	switch memtype {
//	default:
//		Panic("makechan: illegal memtype:", memtype)
//	case CPUMemory:
//		c.slice = Float32ToSlice(make([]float32, N))
//	case GPUMemory:
//		c.slice = gpuSlice(N)
//	case UnifiedMemory:
//		c.slice = unifiedSlice(N)
//	}
//	return c
//}
//
//// UnsafeData returns the underlying storage without locking.
//// Intended only for page-locking, not for reading or writing.
//func (d *chandata) UnsafeData() []float32 { return d.slice.Host() }
//
//func (d *chandata) NComp() int { return len(d.slice.Host()) }
//
//func (q Chan1) String() string {
//	unit := q.unit
//	if unit != "" {
//		unit = " [" + unit + "]"
//	}
//	return fmt.Sprint(q.tag, unit, ": ", q.NComp(), "x", q.Size(), ", ", q.nBlocks, " blocks")
//}
//
//func (c chandata) UnsafeArray() [][][]float32 {
//	return Reshape(c.slice.Host(), c.Size())
//}
