package engine

// TODO: rm

// Returns a constant scalar function. E.g.:
// 	Alpha = Const(1) // time-independent
func Const(value float64) func() float64 {
	return func() float64 { return value }
}

// Returns a constant vector function. E.g.:
// 	B_ext = ConstVector(1e-3, 0, 0) // 1mT along X
func ConstVector(x, y, z float64) func() [3]float64 {
	return func() [3]float64 { return [3]float64{x, y, z} }
}
