package mag

// Utilities for setting magnetic configurations.

import (
"nimble-cube/core"
)

// Initializes array uniformly to the vector value.
func Uniform(array [3][][][]float32, value Vector) {
	for c,comp := range array{
		a := core.Contiguous(comp)
		for i:= range a{
			a[i]=value[c]
		}
	}
}
