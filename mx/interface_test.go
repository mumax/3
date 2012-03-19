package mx

import (
	"fmt"
	"testing"
)

// test assignment to various interface types
func TestInterface(t *testing.T) { //←[ TestInterface t does not escape]
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
	fmt.Println(QString(quant)) //←[ TestInterface ... argument does not escape]

	uniformN := NewUniform([]float32{2, 3}) //←[ TestInterface []float32 literal does not escape]
	quant = uniformN
	uniform = uniformN
	fmt.Println(QString(uniform)) //←[ TestInterface ... argument does not escape]

	uniform1 := NewUniformScalar(42)
	quant = uniform1
	uniform = uniform1
	scalar = uniform1
	uniformScalar = uniform1
	fmt.Println(QString(uniform)) //←[ TestInterface ... argument does not escape]

	uniform3 := NewUniformVector([3]float32{1, 2, 3})
	quant = uniform3
	uniform = uniform3
	vector = uniform3
	uniformVector = uniform3
	fmt.Println(QString(uniform)) //←[ TestInterface ... argument does not escape]
}
