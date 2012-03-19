package mx

import (
	"fmt"
	"testing"
)

func TestInterface(t *testing.T) {
	var quant Quant
	var scalar Scalar
	var vector Vector
	var uniform Uniform
	var uniformScalar UniformScalar
	var uniformVector UniformVector

	quant = scalar
	quant = vector
	quant = uniform
	quant = uniformScalar
	quant = uniformVector

	fmt.Println(QString(quant))
}
