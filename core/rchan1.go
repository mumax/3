package core

// Read-only Chan.
type RChan1 struct {
	chandata
	mutex *RMutex
}

func (c *Chan1) NewReader() RChan1 {
	return RChan1{c.chandata, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan1) ReadNext(n int) []float32 {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.slice.Slice(a, b).list
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *RChan1) ReadDone() {
	c.mutex.ReadDone()
}

func (c *RChan1) ReadDelta(Δstart, Δstop int) []float32 {
	c.mutex.ReadDelta(Δstart, Δstop)
	a, b := c.mutex.RRange()
	return c.slice.Slice(a, b).list
}

