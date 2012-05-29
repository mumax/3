package nc

// Set all elements to value.
func Memset(array []float32, value float32) {
	for i := range array {
		array[i] = value
	}
}
