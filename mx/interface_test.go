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

	uniformN := NewUniform([]float32{2, 3, 4})
	quant = uniformN
	uniform = uniformN
	fmt.Println(QString(uniform))

	uniform1 := NewUniformScalar(42)
	quant = uniform1
	uniform = uniform1
	scalar = uniform1
	uniformScalar = uniform1
	fmt.Println(QString(uniform))

}
