package nc

import (
	"math"
)

// 3-component vector.
type Vector [3]float32

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

func (a Vector) Scale(s float32) Vector {
	return Vector{s * a[X], s * a[Y], s * a[Z]}
}
