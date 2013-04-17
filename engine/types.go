package engine

// Scalar function, used e.g. to set material parameters.
type ScalFn func() float64

// Vector function, used e.g. to set material parameters.
type VecFn func() [3]float64

// Returns a constant scalar function.
func Const(value float64) ScalFn {
	return ScalFn(func() float64 { return value })
}

// Returns a constant vector function.
func ConstVector(x, y, z float64) VecFn {
	return VecFn(func() [3]float64 { return [3]float64{x, y, z} })
}
