package main

import "math"

// available windowing functions
var windows = map[string]windowFunc{
	"boxcar":  boxcar,
	"hamming": hamming,
	"hann":    hann,
	"welch":   welch,
}

// returns weight for element n in array of N
type windowFunc func(n, N float32) float32

// multiply all elements by window functions
func applyWindow(data []float32, window windowFunc) {
	N := float32(len(data))
	for i := range data {
		n := float32(i) / N
		data[i] *= window(n, N)
	}
}

func boxcar(n, N float32) float32 {
	return 1
}

func welch(n, N float32) float32 {
	return 1 - sqr((n-(N-1)/2)/((N-1)/2))
}

func hann(n, N float32) float32 {
	return 0.5 * (1 + cos((2*math.Pi*n)/(N-1)))
}

func hamming(n, N float32) float32 {
	const a = 0.54
	const b = 1 - a
	return a + b*cos((2*math.Pi*n)/(N-1))
}

func sqr(x float32) float32 { return x * x }
func cos(x float32) float32 { return float32(math.Cos(float64(x))) }
