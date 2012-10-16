package core

type Chan struct {
	chandata // array+list
	mutex    *RWMutex
}

type chandata struct {
	array [][][]float32
	list  []float32
}

func MakeChan(size [3]int, tag string) Chan {
	return Chan{makedata(size), NewRWMutex(Prod(size), tag)}
}

// Implements Chans
func (c *Chan) Chans() []Chan {
	return []Chan{*c}
}

func makedata(size [3]int) chandata {
	var c chandata
	c.array = MakeFloats(size)
	c.list = Contiguous(c.array)
	return c
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan.
// When done, WriteDone() should be called to "send" the
// slice down the Chan. After that, the slice is not valid any more.
func (c *Chan) WriteNext(n int) []float32 {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return c.list[a:b]
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *Chan) WriteDone() {
	c.mutex.WriteDone()
}

func (c *Chan) WriteDelta(Δstart, Δstop int) []float32 {
	c.mutex.WriteDelta(Δstart, Δstop)
	a, b := c.mutex.WRange()
	return c.list[a:b]
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chandata) UnsafeData() []float32 {
	return d.list
}

func (d *chandata) UnsafeArray() [][][]float32 {
	return d.array
}

func (d *chandata) Size() [3]int {
	return SizeOf(d.array)
}
