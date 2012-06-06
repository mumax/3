package mm

// TODO: mv->garbageman.go
func Buffer() []float32 {
	return make([]float32, warp) //<-take
}
func Buffer3() [3][]float32 {
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}
