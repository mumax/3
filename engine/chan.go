package engine

// File: Chan stores a physical quantity.
// Author: Arne Vansteenkiste

import (
	"code.google.com/p/mx3/data"
	"log"
)

// shared by Chan and Reader
type chan_ struct {
	buffer data.Slice           // stores the data
	lock   [data.MAX_COMP]mutex // protects buffer. mutex can be read or write type TODO: make slice, also for Slice
}

type Chan struct {
	chan_
}

func ChanFromSlice(s *data.Slice) *Chan {
	N := s.Mesh().NCell()
	var lock [data.MAX_COMP]mutex
	nComp := s.NComp()
	for c := 0; c < nComp; c++ {
		lock[c] = newRWMutex(N)
	}
	return &Chan{chan_{*s, lock}}
}

type Reader struct {
	chan_
}

func (c *Chan) NewReader() *Reader {
	reader := (*c).chan_ // copy
	for i := range reader.lock {
		reader.lock[i] = (reader.lock[i]).(*rwMutex).MakeRMutex()
	}
	return &Reader{reader}
}

// NComp returns the number of components.
// 	scalar: 1
// 	vector: 3
// 	...
func (c *chan_) NComp() int {
	return int(c.buffer.NComp())
}

func (c *chan_) comp(i int) chan_ {
	return chan_{*c.buffer.Comp(i), [data.MAX_COMP]mutex{c.lock[i]}}
}

func (c *Chan) Comp(i int) Chan {
	return Chan{c.comp(i)}
}

func (c *Reader) Comp(i int) Reader {
	return Reader{c.comp(i)}
}

func (c *chan_) Mesh() *data.Mesh {
	return c.buffer.Mesh()
}

// Returns the data buffer without locking.
// To be used with extreme care.
func (c *chan_) Data() *data.Slice {
	for i := 0; i < c.NComp(); i++ {
		if c.lock[i].isLocked() {
			log.Panic("chan unsafe data: is locked")
		}
	}
	return &c.buffer
}

// lock the next n elements of buffer.
func (c *chan_) next() *data.Slice {
	//n := c.Mesh().NCell()
	c.lock[0].lockNext()
	a, b := c.lock[0].lockedRange()
	ncomp := c.NComp()
	for i := 1; i < ncomp; i++ {
		c.lock[i].lockNext()
		α, β := c.lock[i].lockedRange()
		if α != a || β != b {
			panic("chan: next: inconsistent state")
		}
	}
	return c.buffer.Slice(a, b)
}

// unlock the locked buffer range.
func (c *chan_) done() {
	for i := 0; i < c.NComp(); i++ {
		c.lock[i].unlock()
	}
}

func (c *Chan) WriteNext() *data.Slice {
	return c.chan_.next()
}

func (c *Chan) WriteDone() {
	c.chan_.done()
}

func (c *Reader) ReadNext() *data.Slice {
	return c.chan_.next()
}

func (c *Reader) ReadDone() {
	c.chan_.done()
}
