package core

// Read-only Chan3.
type RChan3 [3]RChan1

func (c *Chan3) NewReader() RChan3 {
	return RChan3{c[0].NewReader(), c[1].NewReader(), c[2].NewReader()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan3.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan3) ReadNext(n int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].mutex.ReadNext(n)
		a, b := c[i].mutex.RRange()
		next[i] = c[i].list[a:b]
	}
	return next
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *RChan3) ReadDone() {
	for i := range c {
		c[i].ReadDone()
	}
}

func (c *RChan3) ReadDelta(Δstart, Δstop int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].mutex.ReadDelta(Δstart, Δstop)
		a, b := c[i].mutex.RRange()
		next[i] = c[i].list[a:b]
	}
	return next
}

func (c *RChan3) Mesh() *Mesh  { return c[0].Mesh }
func (c *RChan3) Size() [3]int { return c[0].Size() }
func (c *RChan3) Unit() string { return c[0].Unit() }
func (c *RChan3) Tag() string  { return c[0].Tag() }

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c *RChan3) UnsafeData() [3][]float32 {
	return [3][]float32{c[0].list, c[1].list, c[2].list}
}
