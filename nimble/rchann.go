package nimble

// Read-only N-component Chan.
type RChanN []RChan1

func (c ChanN) NewReader() RChanN {
	return RChanN{c.comp[0].NewReader(), c.comp[1].NewReader(), c.comp[2].NewReader()}
}

func (c RChanN) Mesh() *Mesh       { return c[0].Mesh }
func (c RChanN) Size() [3]int      { return c[0].Size() }
func (c RChanN) Unit() string      { return c[0].Unit() }
func (c RChanN) Tag() string       { return c[0].Tag() }
func (c RChanN) NComp() int        { return len(c) }
func (c RChanN) Comp(i int) RChan1 { return c[i] }
func (c RChanN) MemType() MemType { return c[0].slice.MemType}

// ReadNext locks and returns a slice of length n for 
// reading the next n elements from the Chan.
// When done, ReadDone() should be called .
// After that, the slice is not valid any more.
func (c RChanN) ReadNext(n int) []Slice {
	next := make([]Slice, c.NComp())
	for i := range c {
		c[i].mutex.ReadNext(n)
		a, b := c[i].mutex.RRange()
		next[i] = c[i].slice.Slice(a, b)
	}
	return next
}

// ReadDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan3.
func (c RChanN) ReadDone() {
	for i := range c {
		c[i].ReadDone()
	}
}

func (c RChanN) UnsafeData() []Slice {
	s := make([]Slice, c.NComp())
	for i := range s {
		s[i] = c[i].slice
	}
	return s
}
