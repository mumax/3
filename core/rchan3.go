package core

// Read-only Chan3.
type RChan3 struct {
	chan3data
	mutex *RMutex
}

func (c *Chan3) MakeRChan3() RChan3 {
	return RChan3{c.chan3data, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan3.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan3) ReadNext(n int) [3][]float32 {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return [3][]float32{c.list[0][a:b], c.list[1][a:b], c.list[2][a:b]}
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c *RChan3) ReadDone() {
	c.mutex.ReadDone()
}
