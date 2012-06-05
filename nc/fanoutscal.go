package nc

// Receive-only side of a FanInScal.
type FanoutScal <-chan float32

// Receive operator.
func (r FanoutScal) Recv() float32 {
	return <-r
}
