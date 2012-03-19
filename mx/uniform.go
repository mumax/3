package mx

import ()

// Uniform quantity is uniform over space.
type Uniform interface {
	Quant
	Get(comp int) float32
}
