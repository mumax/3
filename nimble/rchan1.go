package nimble

// Read-only Chan.
type RChan1 struct {
	*Info
	slice Slice // TODO: rename buffer
	mutex *rMutex
}

func (c Chan1) NewReader() RChan1 {
	return RChan1{c.Info, c.slice, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c RChan1) ReadNext(n int) Slice {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.slice.Slice(a, b)
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c RChan1) ReadDone() {
	c.mutex.ReadDone()
}

//func (c *RChan1) ReadDelta(Δstart, Δstop int) []float32 {
//	c.mutex.ReadDelta(Δstart, Δstop)
//	a, b := c.mutex.RRange()
//	return c.slice.Slice(a, b).list
//}
