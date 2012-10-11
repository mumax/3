package gpu

import (
	"github.com/barnex/cuda5/safe"
	//"nimble-cube/core"
)

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c *Chan3) UnsafeData() [3]safe.Float32s {
	return [3]safe.Float32s{c[0].list, c[1].list, c[2].list}
}

//func (c *Chan3) UnsafeArray() [3][][][]float32 {
//	return [3][][][]float32{c[0].array, c[1].array, c[2].array}
//}

func (c *Chan3) Size() [3]int {
	return c[0].Size()
}

// Chan of 3-vector data.
type Chan3 [3]Chan

func MakeChan3(size [3]int, tag string) Chan3 {
	return Chan3{MakeChan(size, tag+"0"), MakeChan(size, tag+"1"), MakeChan(size, tag+"2")}
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *Chan3) WriteNext(n int) [3]safe.Float32s {
	var next [3]safe.Float32s
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].list.Slice(a, b)
	}
	return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *Chan3) WriteDone() {
	for i := range c {
		c[i].WriteDone()
	}
}

//func (c *Chan3) WriteDelta(Δstart, Δstop int) [3][]float32 {
//	var next [3][]float32
//	for i := range c {
//		c[i].WriteDelta(Δstart, Δstop)
//		a, b := c[i].mutex.WRange()
//		next[i] = c[i].list[a:b]
//	}
//	return next
//}

//
//type Chan3 [3]Chan
//
//func MakeChan3(size [3]int, tag string) Chan3 {
//	return Chan3{MakeChan(size, tag), MakeChan(size, tag), MakeChan(size, tag)}
//}
//
////func (c *Chan3) Vectors() [3]safe.Float32s {
////	return [3]safe.Float32s{c[0].Float32s, c[1].Float32s, c[2].Float32s}
////}
//
////func (c *Chan3) RWMutex() core.RWMutex3 {
////	return core.RWMutex3{c[0].RWMutex, c[1].RWMutex, c[2].RWMutex}
////}
//
//// Read-only Chan3
//type RChan3 [3]RChan
//
//func (c *Chan3) ReadOnly() RChan3 {
//	return RChan3{c[0].ReadOnly(), c[1].ReadOnly(), c[2].ReadOnly()}
//}
//
//
//
////type chan3data struct {
////	list [3]safe.Float32s
////	size [3]int
////}
////
////func (c *chan3data) Size() [3]int { return c.size }
////
////func (c *chan3data) UnsafeData() [3]safe.Float32s { return c.list }
////
////type Chan3 struct {
////	chan3data
////	mutex *core.RWMutex
////}
////
////func MakeChan3(size [3]int, tag string) Chan3 {
////	return Chan3{chan3data{MakeVectors(core.Prod(size)), size}, core.NewRWMutex(core.Prod(size), tag)}
////}
////
////// WriteNext locks and returns a slice of length n for 
////// writing the next n elements to the Chan3.
////// When done, WriteDone() should be called to "send" the
////// slice down the Chan3. After that, the slice is not valid any more.
////func (c *Chan3) WriteNext(n int) [3]safe.Float32s {
////	c.mutex.WriteNext(n)
////	a, b := c.mutex.WRange()
////	return [3]safe.Float32s{c.list[0].Slice(a, b), c.list[1].Slice(a, b), c.list[2].Slice(a, b)}
////}
////
////// WriteDone() signals a slice obtained by WriteNext() is fully
////// written and can be sent down the Chan3.
////func (c *Chan3) WriteDone() {
////	c.mutex.WriteDone()
////}
