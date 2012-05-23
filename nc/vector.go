package nc

import ()

// 3-component vector.
type Vector [3]float32

// Index for vector component x,y,z.
const (
	X = 0
	Y = 1
	Z = 2
)

// Vector addition
func (a Vector) Add(b Vector) Vector {
	return Vector{a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z]}
}

// Vector subtraction
func (a Vector) Sub(b Vector) Vector {
	return Vector{a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
}
