package nc

// Receive-only side of a FanInScal.
type FanoutScalar <-chan float32

// Receive operator.
func (r FanoutScalar) Recv() float32 {
	return <-r
}
