package gpu

//
//import (
//	"github.com/barnex/cuda5/safe"
//	"nimble-cube/core"
//)
//
//// Read-only Chan3.
//type RChanN []core.RChan1
//
//func (c core.ChanN) NewReader() RChanN {
//	r := make(RChanN, c.NComp())
//	for i := range r {
//		r[i] = c[i].NewReader()
//	}
//	return r
//}
//
//// ReadNext locks and returns a slice of length n for 
//// reading the next n elements from the Chan3.
//// When done, ReadDone() should be called .
//// After that, the slice is not valid any more.
//func (c RChanN) ReadNext(n int) []safe.Float32s {
//	next := make([]safe.Float32s, c.NComp())
//	for i := range c {
//		c[i].mutex.ReadNext(n)
//		a, b := c[i].mutex.RRange()
//		next[i] = c[i].gpu.Slice(a, b)
//	}
//	return next
//}
//
//// ReadDone() signals a slice obtained by WriteNext() is fully
//// written and can be sent down the Chan3.
//func (c RChanN) ReadDone() {
//	for i := range c {
//		c[i].ReadDone()
//	}
//}
//
//func (c RChanN) ReadDelta(Δstart, Δstop int) [3]safe.Float32s {
//	var next [3]safe.Float32s
//	for i := range c {
//		c[i].mutex.ReadDelta(Δstart, Δstop)
//		a, b := c[i].mutex.RRange()
//		next[i] = c[i].gpu.Slice(a, b)
//	}
//	return next
//}
//
//func (c RChanN) Size() [3]int     { return c[0].Size() }
//func (c RChanN) NComp() int       { return len(c) }
//func (c RChanN) Unit() string     { return c[0].Unit() }
//func (c RChanN) Mesh() *core.Mesh { return c[0].Mesh }
//
//// UnsafeData returns the underlying storage without locking.
//// Intended only for page-locking, not for reading or writing.
////func (c *RChanN) UnsafeData() [3]safe.Float32s {
////	return [3]safe.Float32s{c[0].list, c[1].list, c[2].list}
////}
