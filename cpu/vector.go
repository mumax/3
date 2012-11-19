package cpu

import (
	"math"
)

// 3-component vector.
type Vector [3]float32

// Vector addition.
func (a Vector) Add(b Vector) Vector {
	return Vector{a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z]}
}

// Multiply-add.
func (a Vector) MAdd(s float32, b Vector) Vector {
	return Vector{a[X] + s*b[X], a[Y] + s*b[Y], a[Z] + s*b[Z]}
}

// Vector subtraction.
func (a Vector) Sub(b Vector) Vector {
	return Vector{a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
}

func (a Vector) Cross(b Vector) Vector {
	return Vector{a[Y]*b[Z] - a[Z]*b[Y], -a[X]*b[Z] + a[Z]*b[X], a[X]*b[Y] - a[Y]*b[X]}
}

// Vector norm.
func (a Vector) Norm() float32 {
	return float32(math.Sqrt(float64(a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z])))
}

// Vector norm squared.
func (a Vector) Norm2() float32 {
	return a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z]
}

// Normalized vector.
func (a Vector) Normalized() Vector {
	norm := a.Norm()
	if norm == 0 {
		return Vector{0, 0, 0}
	}
	inorm := 1 / norm
	return Vector{inorm * a[X], inorm * a[Y], inorm * a[Z]}
}

// Scalar x Vector product.
func (a Vector) Scaled(s float32) Vector {
	return Vector{s * a[X], s * a[Y], s * a[Z]}
}

const(
	X=0
	Y=1
	Z=2
)
