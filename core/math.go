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

func Average(x []float32) float64 {
	sum := 0.
	for _, v := range x {
		sum += float64(v)
	}
	return sum / float64(len(x))
}
