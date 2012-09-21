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

type Chan3 [3]Chan

func MakeChan3(size [3]int) Chan3 {
	return Chan3{MakeChan(size), MakeChan(size), MakeChan(size)}
}

func (c *Chan3) List() [3][]float32 {
	return [3][]float32{c[0].List, c[1].List, c[2].List}
}

func (c *Chan3) Array() [3][][][]float32 {
	return [3][][][]float32{c[0].Array, c[1].Array, c[2].Array}
}

func (c *Chan3) RWMutex() RWMutex3 {
	return RWMutex3{c[0].RWMutex, c[1].RWMutex, c[2].RWMutex}
}
