package core

type Chan struct {
	Array    [][][]float32
	List     []float32
	*RWMutex // TODO: hide NewReader()?, add WChan (write-only)?
}

func MakeChan(size [3]int) Chan {
	var c Chan
	c.Array = MakeFloats(size)
	c.List = Contiguous(c.Array)
	c.RWMutex = NewRWMutex(Prod(size))
	return c
}

type RChan struct {
	Array [][][]float32
	List  []float32
	*RMutex
}

func (c *Chan) ReadOnly() RChan {
	return RChan{c.Array, c.List, c.RWMutex.NewReader()}
}
