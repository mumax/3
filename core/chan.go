package core

type chandata struct {
	array [][][]float32
	list  []float32
}

func makedata(size [3]int) chandata {
	var c chandata
	c.array = MakeFloats(size)
	c.list = Contiguous(c.array)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chandata) UnsafeData() []float32 {
	return d.list
}

func (d *chandata) Size() [3]int {
	return SizeOf(d.array)
}

type Chan struct {
	chandata // array+list
	mutex    *RWMutex
}

func MakeChan(size [3]int) Chan {
	return Chan{makedata(size), NewRWMutex(Prod(size))}
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
