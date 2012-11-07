package nimble

// Chan of 3-vector data.
type Chan3 ChanN

func MakeChan3(tag, unit string, m *Mesh, memType MemType, blocks ...int) Chan3 {
	return Chan3(MakeChanN(3, tag, unit, m, memType, blocks...))
}

func (c Chan3) WriteNext(n int) [3][]float32 {
	next := ChanN(c).WriteNext(n)
	return [3][]float32{next[0].Host(), next[1].Host(), next[2].Host()}
}

//func (c Chan3) WriteDelta(Δstart, Δstop int) [3][]float32 {
//	next := ChanN(c).WriteDelta(Δstart, Δstop)
//	return [3][]float32{next[0], next[1], next[2]}
//}

func (c Chan3) Mesh() *Mesh  { return ChanN(c).Mesh() }
func (c Chan3) Size() [3]int { return ChanN(c).Size() }
func (c Chan3) WriteDone()   { ChanN(c).WriteDone() }
func (c Chan3) Unit() string { return ChanN(c).Unit() }
func (c Chan3) Tag() string  { return ChanN(c).Tag() }
func (c Chan3) ChanN() ChanN { return ChanN(c) }
func (c Chan3) NComp() int   { return ChanN(c).NComp() }

//func (c Chan3) UnsafeData() [3][]float32 {
//	return [3][]float32{c[0].slice.Host(), c[1].slice.Host(), c[2].slice.Host()}
//}

//func (c Chan3) UnsafeArray() [3][][][]float32 {
//	return [3][][][]float32{c[0].UnsafeArray(), c[1].UnsafeArray(), c[2].UnsafeArray()}
//}
