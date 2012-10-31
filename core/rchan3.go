package core

// Read-only Chan3.
type RChan3 RChanN

func (c Chan3) NewReader() RChan3 {
	return RChan3{c[0].NewReader(), c[1].NewReader(), c[2].NewReader()}
}

func (c RChan3) ReadNext(n int) [3][]float32 {
	next := RChanN(c).ReadNextList(n)
	return [3][]float32{next[0], next[1], next[2]}
}

func (c RChan3) ReadDelta(Δstart, Δstop int) [3][]float32 {
	next := RChanN(c).ReadDelta(Δstart, Δstop)
	return [3][]float32{next[0], next[1], next[2]}
}

func (c RChan3) ReadDone()    { RChanN(c).ReadDone() }
func (c RChan3) Mesh() *Mesh  { return RChanN(c).Mesh() }
func (c RChan3) Unit() string { return RChanN(c).Unit() }
func (c RChan3) Tag() string  { return RChanN(c).Tag() }
func (c RChan3) NComp() int   { return len(c) }

func (c RChan3) UnsafeData() [3][]float32 {
	return [3][]float32{c[0].slice.list, c[1].slice.list, c[2].slice.list}
}
