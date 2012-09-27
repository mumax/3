package core

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (c *Chan3) UnsafeData() [3][]float32 {
	return [3][]float32{c[0].list, c[1].list, c[2].list}
}

func (c *Chan3) UnsafeArray() [3][][][]float32 {
	return [3][][][]float32{c[0].array, c[1].array, c[2].array}
}

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
func (c *Chan3) WriteNext(n int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].WriteNext(n)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].list[a:b]
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

func (c *Chan3) WriteDelta(Δstart, Δstop int) [3][]float32 {
	var next [3][]float32
	for i := range c {
		c[i].WriteDelta(Δstart, Δstop)
		a, b := c[i].mutex.WRange()
		next[i] = c[i].list[a:b]
	}
	return next
}
