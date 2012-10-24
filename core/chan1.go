package core

type Chan1 struct {
	chandata // array+list
	mutex    *RWMutex
}

type chandata struct {
	*Info
	array [][][]float32
	list  []float32
}

func MakeChan(tag, unit string, m *Mesh, blocks ...int) Chan1 {
	tag = UniqueTag(tag)
	data := makedata(tag, unit, m, blocks...)
	return Chan1{data, NewRWMutex(data.BlockLen(), tag)}
}

// Implements Chans
func (c *Chan1) Chan() []Chan1 {
	return []Chan1{*c}
}

func makedata(tag, unit string, m*Mesh, blocks...int) chandata {
	size := m.Size()
	var c chandata
	c.Info = NewInfo(tag, unit, m, blocks...)

	c.array = MakeFloats(size)
	c.list = Contiguous(c.array)
	return c
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan.
// When done, WriteDone() should be called to "send" the
// slice down the Chan. After that, the slice is not valid any more.
func (c *Chan1) WriteNext(n int) []float32 {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return c.list[a:b]
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *Chan1) WriteDone() {
	c.mutex.WriteDone()
}

func (c *Chan1) WriteDelta(Δstart, Δstop int) []float32 {
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
