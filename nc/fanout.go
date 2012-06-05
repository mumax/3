package nc

// Receive-only side of a FanIn.
type FanOut <-chan []float32

// Receive operator.
func (r FanOut) Recv() []float32 {
	return <-r
}
