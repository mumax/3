package nc

// Receive-only endpoint of a FanIn3.
type FanOut3 [3]<-chan []float32

// Vector receive operator.
func (v *FanOut3) Recv() [3][]float32 {
	return [3][]float32{<-v[X], <-v[Y], <-v[Z]}
}

func (v *FanOut3) IsNil() bool {
	return v[X] == nil
}
