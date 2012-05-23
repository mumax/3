package nc

import (
)

type Vector [3]float32

const (
	X = 0
	Y = 1
	Z = 2
)

func (a Vector) Add(b Vector) Vector {
	return Vector{a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z]}
}

func (a Vector) Sub(b Vector) Vector {
	return Vector{a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]}
}
