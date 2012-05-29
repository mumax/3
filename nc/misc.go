package nc

// Set all elements to value.
func Memset(array []float32, value float32) {
	for i := range array {
		array[i] = value
	}
}

// Check if all sizes are > 0
func checkSize(size []int) {
	for i, s := range size {
		if s < 1 {
			Panic("MakeBlock: N", i, " out of range:", s)
		}
	}
}
