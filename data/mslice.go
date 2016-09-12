package data

// Slice + scalar multiplier.
type MSlice struct {
	arr *Slice
	mul float64
}

func (m MSlice) Size() [3]int {
	return m.arr.Size()
}
