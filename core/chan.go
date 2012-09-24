package core

type data struct {
	array [][][]float32
	list  []float32
}

func makedata(size [3]int) data {
	var c data
	c.array = MakeFloats(size)
	c.list = Contiguous(c.array)
	return c
}

// UnsafeData returns the underlying storage without locking.
// Intended only for page-locking, not for reading or writing.
func(d*data) UnsafeData()[]float32{
	return d.list
}

type Chan struct {
	data  // array+list
	mutex *RWMutex
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

//
//// Chan of vector data
//type Chan3 [3]Chan
//
//func MakeChan3(size [3]int) Chan3 {
//	return Chan3{MakeChan(size), MakeChan(size), MakeChan(size)}
//}
//
//func (c *Chan3) List() [3][]float32 {
//	return [3][]float32{c[0].List, c[1].List, c[2].List}
//}
//
//func (c *Chan3) Array() [3][][][]float32 {
//	return [3][][][]float32{c[0].Array, c[1].Array, c[2].Array}
//}
//
//func (c *Chan3) RWMutex() RWMutex3 {
//	return RWMutex3{c[0].RWMutex, c[1].RWMutex, c[2].RWMutex}
//}
//
//// Read-only Chan3
//type RChan3 [3]RChan
//
//func (c *Chan3) ReadOnly() RChan3 {
//	return RChan3{c[0].ReadOnly(), c[1].ReadOnly(), c[2].ReadOnly()}
//}
//
//func (c *RChan3) ReadNext(delta int) {
//	for i := range c {
//		c[i].ReadNext(delta)
//	}
//}
//
//func (c *RChan3) ReadDone() {
//	for i := range c {
//		c[i].ReadDone()
//	}
//}
//
//func (c *Chan3) WriteNext(delta int) {
//	for i := range c {
//		c[i].WriteNext(delta)
//	}
//}
//
//func (c *Chan3) WriteDone() {
//	for i := range c {
//		c[i].WriteDone()
//	}
//}
//
//func (c *RChan3) List() [3][]float32 {
//	return [3][]float32{c[0].List, c[1].List, c[2].List}
//}
//
//func (c *RChan3) Array() [3][][][]float32 {
//	return [3][][][]float32{c[0].Array, c[1].Array, c[2].Array}
//}
