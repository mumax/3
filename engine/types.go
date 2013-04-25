package engine

// Scalar function, used to set time-dependent parameters. E.g.:
// 	Alpha = func()float64{
// 		if Time < 1e-9 {
// 			return 1
// 		}else{
// 			return 0.01
// 		}
// 	}
type ScalFn func() float64

// Vector function, used to set time-dependent parameters. E.g.:
// 	// Rotating magnetic field
// 	B_ext = func()[3]float64{
// 		bx := 1e-3 * math.Sin(2*math.Pi * freq * Time)
// 		by := 1e-3 * math.Cos(2*math.Pi * freq * Time)
// 		return [3]float64{bx, by, 0}
// 	}
type VecFn func() [3]float64

// Returns a constant scalar function. E.g.:
// 	Alpha = Const(1) // time-independent
func Const(value float64) ScalFn {
	return ScalFn(func() float64 { return value })
}

// Returns a constant vector function. E.g.:
// 	B_ext = ConstVector(1e-3, 0, 0) // 1mT along X
func ConstVector(x, y, z float64) VecFn {
	return VecFn(func() [3]float64 { return [3]float64{x, y, z} })
}
