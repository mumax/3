package mx

import ()

type Vector interface {
	Quant
	IGet3(index int) [3]float32
}

type UniformVector interface {
	Quant
	IGet3(index int) [3]float32
}
