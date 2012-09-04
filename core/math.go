package core

import ()

// dst = a+b.
func Add(dst, a, b []float32) {
	Assert(len(dst) == len(a) && len(a) == len(b))
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// dst = a+b.
func Add3(dst, a, b [3][]float32) {
	for c := range dst {
		Add(dst[c], a[c], b[c])
	}
}
