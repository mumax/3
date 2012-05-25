package nc

import (
	"math"
)

// 3-component vector.
type Vector [VECCOMP]float32

// Number of vector components.
const VECCOMP = 3

// Index for vector component x,y,z.
const (
	X = 0
	Y = 1
	Z = 2
)

// Vector addition.
func (a Vector) Add(b Vector) Vector {
	return Vector{a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z]}
}

// Vector subtraction.
func (a Vector) Sub(b Vector) Vector {
	return Vector{a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
}

// Vector norm.
func (a Vector) Norm() float32 {
	return float32(math.Sqrt(float64(a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z])))
}

// Vector norm squared.
func (a Vector) Norm2() float32 {
	return a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z]
}
