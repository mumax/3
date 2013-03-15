package engine

type Scal interface {
	Val(time float64) float64
}

type Vec interface {
	Val(time float64) [3]float64
}

func Const(value float64) Scal {
	return scalFn(func(time float64) float64 { return value })
}

func Vector(x, y, z float64) Vec {
	return vecFn(func(time float64) [3]float64 { return [3]float64{x, y, z} })
}

type scalFn func(time float64) float64

func (f scalFn) Val(time float64) float64 { return f(time) }

type vecFn func(time float64) [3]float64

func (f vecFn) Val(time float64) [3]float64 { return f(time) }
