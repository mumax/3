package nimble

// Constant continuously pipes a constant array into a channel.
type Constant struct {
	output ChanN
}

func NewConstant(tag, unit string, m *Mesh, data []Slice) *Constant {
	c := new(Constant)
	c.output = AsChan(data, tag, unit, m)
	Stack(c)
	return c
}

func (c *Constant) Run() {
	for {
		c.output.WriteNext(c.output.BufLen())
		c.output.WriteDone()
	}
}

func (c *Constant) Output() ChanN {
	return c.output
}
