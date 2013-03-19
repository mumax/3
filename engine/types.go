package engine

type ScalFn func() float64

type VecFn func() [3]float64

func Const(value float64) ScalFn {
	return ScalFn(func() float64 { return value })
}

func ConstVector(x, y, z float64) VecFn {
	return VecFn(func() [3]float64 { return [3]float64{x, y, z} })
}
