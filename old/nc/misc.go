package nc

// Set all elements to value.
func Memset(array []float32, value float32) {
	for i := range array {
		array[i] = value
	}
}

// Set all elements to vector.
func Memset3(array [3][]float32, value Vector) {
	for i, a := range array {
		Memset(a, value[i])
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
