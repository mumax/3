package nimble

// WChan1

// Chan1 is a Chan that passes 1-component float32 data.
type Chan1 struct {
	*chan1
}

// TODO: Idea: tag, unit are *string, optional and defaulted to unique value
// can be set by user if wanted.
func MakeChan1(tag, unit string, m *Mesh, memType MemType, bufBlocks int) Chan1 {
	N := -666
	if bufBlocks < 1 { // means auto
		N = m.NCell() // buffer all
	} else {
		N = m.BlockLen() * bufBlocks
	}
	return Chan1{aschan1(MakeSlice(N, memType), tag, unit, m, newRWMutex(N))}
}

func (c Chan1) ChanN() ChanN { return ChanN{[]Chan1{c}} }

// WriteDone() signals a slice obtained by WriteNext() is done written.
func (c Chan1) WriteDone() { c.unlock() }

// WriteNext returns a buffer Slice of length n to which data
// can be written. Should be followed by ReadDone().
func (c Chan1) WriteNext(n int) Slice {
	c.lockNext(n)
	a, b := c.lockedRange()
	return c.buffer.Slice(a, b)
}

// NComp returns the number of components (1: scalar, 3: vector, ...)
func (c Chan1) NComp() int { return 1 }

// BufLen returns the largest buffer size n that can be obained
// with ReadNext/WriteNext.
func (c Chan1) BufLen() int { return c.buffer.Len() }

func (c Chan1) NBufferedBlocks() int { return idiv(c.NCell(), c.buffer.Len()) }

//func (c *Chan1) WriteDelta(Δstart, Δstop int) []float32 {
//	c.mutex.WriteDelta(Δstart, Δstop)
//	a, b := c.mutex.WRange()
//	return c.slice.list[a:b]
//}
