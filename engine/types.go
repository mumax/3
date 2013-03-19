package engine

import (
	"code.google.com/p/mx3/data"
)

type ScalFn func() float64

type VecFn func() [3]float64

func Const(value float64) ScalFn {
	return ScalFn(func() float64 { return value })
}

func ConstVector(x, y, z float64) VecFn {
	return VecFn(func() [3]float64 { return [3]float64{x, y, z} })
}

type Quant struct {
	addTo func(dst *data.Slice) // adds quantity to dst
}

func (q *Quant) AddTo(dst *data.Slice) {
	// if need output:
	// add to zeroed buffer, output buffer (async), add buffer to dst
	// pipe buffers to/from output goroutine
	q.addTo(dst)
}
