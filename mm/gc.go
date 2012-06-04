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

func Buffer() []float32       { return <-take }
func VecBuffer() [3][]float32 { return [3][]float32{<-take, <-take, <-take} }
