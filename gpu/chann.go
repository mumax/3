package gpu

import (
	"fmt"
	"github.com/barnex/cuda5/safe"
	"nimble-cube/core"
)

// Chan of arbitrary component data.
type ChanN []Chan1

func MakeChanN(nComp int, tag, unit string, m *core.Mesh, blocks ...int) ChanN {
	tag = core.UniqueTag(tag)
	c := make(ChanN, nComp)
	for i := range c {
		c[i] = MakeChan1(fmt.Sprint(tag, i), unit, m, blocks...)
	}
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
//func (c *ChanN) UnsafeData() []safe.Float32s {
//	return [3]safe.Float32s{c[0].list, c[1].list, c[2].list}
//}

func (c ChanN) Mesh() *core.Mesh { return c[0].Mesh }
func (c ChanN) NComp() int       { return len(c) }
func (c ChanN) Comp(i int) Chan1 { return c[i] }
func (c ChanN) Size() [3]int     { return c[0].Size() }
func (c ChanN) NBlocks() int     { return c[0].NBlocks() }
func (c ChanN) BlockLen() int    { return c[0].BlockLen() }
func (c ChanN) Unit() string     { return c[0].Unit() }

// TODO
func (c ChanN) Tag() string { return c[0].Tag() }

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the ChanN.
// When done, WriteDone() should be called to "send" the
// slice down the ChanN. After that, the slice is not valid any more.
func (c ChanN) WriteNext(n int) [3]safe.Float32s {
	var next [3]safe.Float32s
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].list.Slice(a, b)
	}
	return next
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the ChanN.
func (c ChanN) WriteDone() {
	for i := range c {
		c[i].WriteDone()
	}
}
