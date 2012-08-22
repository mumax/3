package conv

// Interface of any convolution.
type Conv interface {
	Input() [3][][][]float32     // Input data
	Output() [3][][][]float32    // Output data
	Kernel() [3][3][][][]float32 // Convolution kernel
	Exec()                       // Executes the convolution
}

// Test if the convolution gives the same result as the brute-force implementation.
func Test(c Conv) {

}
