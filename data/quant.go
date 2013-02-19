package data

// File: Quant stores a physical quantity.
// Author: Arne Vansteenkiste

// shared by Quant and Reader
type quant struct {
	buffer Slice           // stores the data
	lock   [MAX_COMP]mutex // protects buffer. mutex can be read or write type
}

type Quant struct {
	quant
}

func QuantFromSlice(s *Slice) *Quant {
	N := s.Mesh().NCell() //bufferSize(m, bufBlocks)
	var lock [MAX_COMP]mutex
	nComp := s.NComp()
	for c := 0; c < nComp; c++ {
		lock[c] = newRWMutex(N)
	}
	return &Quant{quant{*s, lock}}
}

func (q *quant) NComp() int {
	return q.buffer.NComp()
}

func (q *Quant) Data() *Slice {
	// fail-fast test, likely to spot abuse sooner or later.
	for c := 0; c < q.NComp(); c++ {
		if q.lock[c].isLocked() {
			panic("quant data is locked")
		}
	}
	return &q.buffer
}

type Reader struct {
	quant
}

func (c *Quant) NewReader() *Reader {
	reader := (*c).quant // copy
	for i := range reader.lock {
		reader.lock[i] = (reader.lock[i]).(*rwMutex).MakeRMutex()
	}
	return &Reader{reader}
}

//// NComp returns the number of components.
//// 	scalar: 1
//// 	vector: 3
//// 	...
//func (c *quant) NComp() int {
//	return int(c.buffer.nComp)
//}
//
//func (c *quant) comp(i int) quant {
//	return quant{c.buffer.Comp(i), [MAX_COMP]mutex{c.lock[i]}, c.tag, c.unit, c.mesh}
//}
//
//func (c *Quant) Comp(i int) Quant {
//	return Quant{c.comp(i)}
//}
//
//func (c *Reader) Comp(i int) Reader {
//	return Reader{c.comp(i)}
//}
//
//// BufLen returns the number of buffered elements.
//// This is the largest number of elements that can be read/written at once.
//func (c *quant) BufLen() int {
//	return c.buffer.Len()
//}
//
//func (c *quant) Unit() string {
//	return c.unit
//}
//
//func (c *quant) Tag() string {
//	return c.tag
//}
//
//func (c *quant) Mesh() *Mesh {
//	return c.mesh
//}
//
//func (c *quant) MemType() MemType {
//	return c.buffer.MemType
//}
//
////func (c Quant) NBufferedBlocks() int { return c.comp[0].NBufferedBlocks() }
////func (c Quant) MemType() MemType { return c.buffer.MemType }
////func (c Quant) Quant() Quant     { return c } // implements Chan iface
////func (c Quant) Comp(i int) Chan1 { return c.comp[i] }
//
//// UnsafeData returns the data buffer without locking.
//// To be used with extreme care.
//func (c *quant) UnsafeData() Slice {
//	return c.buffer
//}
//
// WriteNext locks and returns a slice of length n for
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *quant) next(n int) *Slice {
	c.lock[0].lockNext(n)
	a, b := c.lock[0].lockedRange()
	for i := 1; i < len(c.lock); i++ {
		c.lock[i].lockNext(n)
		α, β := c.lock[i].lockedRange()
		if α != a || β != b {
			panic("chan: next: inconsistent state")
		}
	}
	next := c.buffer.Slice(a, b)
	return next
}

func (c *quant) done() {
	for i := range c.lock {
		c.lock[i].unlock()
	}
}

func (c *Quant) WriteNext(n int) *Slice {
	return c.quant.next(n)
}

func (c *Quant) WriteDone() {
	c.quant.done()
}

func (c *Reader) ReadNext(n int) *Slice {
	return c.quant.next(n)
}

func (c *Reader) ReadDone() {
	c.quant.done()
}

//// WriteDone() signals a slice obtained by WriteNext() is fully
//// written and can be sent down the Chan3.
//
////func (c Quant) WriteDelta(Δstart, Δstop int) [][]float32 {
////	next := make([][]float32, c.NComp())
////	for i := range c {
////		c[i].WriteDelta(Δstart, Δstop)
////		a, b := c[i].mutex.WRange()
////		next[i] = c[i].slice.Slice(a, b).list
////	}
////	return next
////}
//
//func bufferSize(m *Mesh, bufBlocks int) int {
//	N := -666
//	if bufBlocks < 1 { // means auto
//		N = m.NCell() // buffer all
//	} else {
//		N = m.BlockLen() * bufBlocks
//	}
//	return N
//}
//
//// safe integer division.
//func idiv(a, b int) int {
//	core.Assert(a%b == 0)
//	return a / b
//}
//
