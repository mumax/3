package core

type chan3data struct {
	array [3][][][]float32
	list  [3][]float32
}

func make3data(size [3]int) chan3data {
	var c chan3data
	c.array = MakeVectors(size)
	c.list = Contiguous3(c.array)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func (d *chan3data) UnsafeData() [3][]float32 {
	return d.list
}

func (d *chan3data) UnsafeArray() [3][][][]float32 {
	return d.array
}

func (d *chan3data) Size() [3]int {
	return SizeOf(d.array[0])
}

type Chan3 struct {
	chan3data // array+list
	mutex     *RWMutex
}

func MakeChan3(size [3]int) Chan3 {
	return Chan3{make3data(size), NewRWMutex(Prod(size))}
}

// WriteNext locks and returns a slice of length n for 
// writing the next n elements to the Chan3.
// When done, WriteDone() should be called to "send" the
// slice down the Chan3. After that, the slice is not valid any more.
func (c *Chan3) WriteNext(n int) [3][]float32 {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return [3][]float32{c.list[0][a:b], c.list[1][a:b], c.list[2][a:b]}
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *Chan3) WriteDone() {
	c.mutex.WriteDone()
}
