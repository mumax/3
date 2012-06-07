package mm

import(."nimble-cube/nc")
type AverageBox struct {
	in  <-chan []float32
	out []chan<- float64
}

func (box *AverageBox) Run() {
	for {
		sum := 0.0
		for I := 0; I < N(); I += WarpLen() {
			in := <-box.in
			for _, value := range in {
				sum += float64(value)
			}
		}
		SendFloat64(box.out, sum/float64(N()))
	}
}
