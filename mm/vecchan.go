package mm // TODO: nc

const (
	X = 0
	Y = 1
	Z = 2
)

type VecChan [3]chan []float32

func MakeVecChan(buf int) VecChan {
	return VecChan{make(chan []float32, buf), make(chan []float32, buf), make(chan []float32, buf)}
}

func (v *VecChan) Send(data [3][]float32) {
	// TODO: select loop so we can send in any order?
	v[X] <- data[X]
	v[Y] <- data[Y]
	v[Z] <- data[Z]
}

func (v *VecChan) Recv() (data [3][]float32) {
	data[X] = <-v[X]
	data[Y] = <-v[Y]
	data[Z] = <-v[Z]
	return
}
