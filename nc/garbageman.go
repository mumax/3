package nc

// Garbageman recycles garbage slices.

// TODO!!

func Buffer() []float32 {
	return make([]float32, WarpLen())
}

func Buffer3() [3][]float32 {
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}
