package core

import ()

// dst[i] = a[i]+b[i].
func Add(dst, a, b []float32) {
	Assert(len(dst) == len(a) && len(a) == len(b))
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// dst[i] = a[i]+b[i] (vectors).
func Add3(dst, a, b [3][]float32) {
	for c := range dst {
		Add(dst[c], a[c], b[c])
	}
}

// dst[i] += cnst.
func AddConst(dst []float32, cnst float32) {
	for i := range dst {
		dst[i] += cnst
	}
}

// Average of x.
func Average(x []float32) float64 {
	sum := 0.
	for _, v := range x {
		sum += float64(v)
	}
	return sum / float64(len(x))
}
