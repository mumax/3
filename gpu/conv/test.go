package conv

// Interface of any convolution.
type Conv interface {
	Input() [3][][][]float32
	Output() [3][][][]float32
	Kernel() [3][3][][][]float32
	Exec()
}

func Test(c Conv) {

}
