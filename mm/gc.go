package mm

var (
	take <-chan []float32
	//giveback chan<-[]float32
)

func RunGC() {
	out := make(chan []float32)
	take = out

	//in := make(chan []float32)
	//giveback = in

	for {
		out <- make([]float32, warp)
	}
}

func Buffer() []float32 {
	return make([]float32, warp) //<-take
}
func VecBuffer() [3][]float32 {
	return [3][]float32{Buffer(), Buffer(), Buffer()}
}
