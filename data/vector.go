package data

import "math"

type Vector [3]float64

func (v Vector) X() float64 { return v[0] }
func (v Vector) Y() float64 { return v[1] }
func (v Vector) Z() float64 { return v[2] }

func (v Vector) Mul(a float64) Vector {
	return Vector{a * v[0], a * v[1], a * v[2]}
}

func (v Vector) Div(a float64) Vector {
	return v.Mul(1 / a)
}

func (a Vector) Add(b Vector) Vector {
	return Vector{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
}

func (a Vector) Sub(b Vector) Vector {
	return Vector{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

func (v Vector) Len() float64 {
	len2 := v.Dot(v)
	return math.Sqrt(len2)
}

func (a Vector) Dot(b Vector) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func (a Vector) Cross(b Vector) Vector {
	x := a[1]*b[2] - a[2]*b[1]
	y := a[2]*b[0] - a[0]*b[2]
	z := a[0]*b[1] - a[1]*b[0]
	return Vector{x, y, z}
}
