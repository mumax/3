package mx

import ()

// Quant represents a physical quantity. 
type Quant interface {
	Name() string
	Unit() string
	NComp() int                                  // Number of components
	IGet(comp int, index1, index2 int) []float32 // Get values of the specified component between index1 (incl.) and index2 (excl).
}

// Uniform quantity is uniform over space.
type Uniform interface {
	Quant
	Get(comp int) float32 // Get the specified component, independently of position.
}

// Scalar quantity has 1 component.
type Scalar interface {
	Quant
	IGet1(i1, i2 int) []float32 // Get the scalar value for this position index.
}

// UniformScalar is uniform over space and has 1 component.
type UniformScalar interface {
	Quant
	Get1() float32 // Get the scalar value, independently of position.
}

// Vector quantity has 3 components.
//type Vector interface {
//Quant
//IGet3() [3][]float32 // Get the vector value for this position index.
//}

// UniformVector is uniform over space and has 3 components.
type UniformVector interface {
	Quant
	Get3() [3]float32 // 
}
