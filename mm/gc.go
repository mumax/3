package mm

var (
	take <-chan []float32
	//giveback chan<-[]float32
)

func RunRecycler() {
	out := make(chan []float32)
	take = out

	//in := make(chan []float32)
	//giveback = in

	for {
		out <- make([]float32, warp)
	}
}
