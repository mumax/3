package nc

import (
	"fmt"
)

type Vector [3]float32

const (
	X = 0
	Y = 1
	Z = 2
)

func (v *Vector) String() string {
	return fmt.Sprint(v[X], ", ", v[Y], ",", v[Z])
}

func (a Vector) Add(b Vector) Vector {
	return Vector{a[X] + b[X], a[Y] + b[Y], a[Z] + b[Z]}
}
