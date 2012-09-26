package core

// Read-only Chan.
type RChan struct {
	chandata
	mutex *RMutex
}

func (c *Chan) MakeRChan() RChan {
	return RChan{c.chandata, c.mutex.MakeRMutex()}
}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c *RChan) ReadNext(n int) []float32 {
	c.mutex.ReadNext(n)
	a, b := c.mutex.RRange()
	return c.list[a:b]
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c *RChan) ReadDone() {
	c.mutex.ReadDone()
}

func (c *RChan) Tag() string {
	return c.mutex.rw.tag
}
