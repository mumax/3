package nimble

type Chan1 struct {
	*Info
	slice Slice // TODO: rename buffer
	mutex *rwMutex
}

func makeChan1(tag, unit string, m *Mesh, memType MemType, blocks ...int) Chan1 {
	AddQuant(tag)
	info:= NewInfo(tag, unit, m, blocks...)
	N := info.BlockLen()
	return Chan1{info, MakeSlice(N, memType), NewRWMutex(N, tag)}
}

// WriteDone() signals a slice obtained by WriteNext() is fully
// written and can be sent down the Chan.
func (c Chan1) WriteDone() {
	c.mutex.WriteDone()
}

func (c Chan1) WriteNext(n int) Slice {
	c.mutex.WriteNext(n)
	a, b := c.mutex.WRange()
	return c.slice.Slice(a, b)
}

func(c Chan1)NComp()int{return 1}



//func (c *Chan1) WriteDelta(Δstart, Δstop int) []float32 {
//	c.mutex.WriteDelta(Δstart, Δstop)
//	a, b := c.mutex.WRange()
//	return c.slice.list[a:b]
//}
