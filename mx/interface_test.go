package mx

import (
	"fmt"
	"testing"
)

// test assignment to various interface types
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

	uniformN := NewUniform([]float32{2, 3})
	quant = uniformN
	uniform = uniformN
	fmt.Println(QString(uniform))

	uniform1 := NewUniformScalar(42)
	quant = uniform1
	uniform = uniform1
	scalar = uniform1
	uniformScalar = uniform1
	fmt.Println(QString(uniform))

	uniform3 := NewUniformVector([3]float32{1, 2, 3})
	quant = uniform3
	uniform = uniform3
	vector = uniform3
	uniformVector = uniform3
	fmt.Println(QString(uniform))
}

// test assignment of List to various interface types
func TestListInterface(t *testing.T) {
	var quant Quant
	var scalar Scalar
	var vector Vector

	list := NewList(3, 100)
	quant = list
	fmt.Println(QString(quant))

	scalarList := NewScalarList(100)
	quant = scalarList
	scalar = scalarList
	fmt.Println(QString(scalar))

	vectorList := NewVectorList(100)
	quant = vectorList
	vector = vectorList
	fmt.Println(QString(vector))
}
