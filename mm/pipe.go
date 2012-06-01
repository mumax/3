package mm // TODO: mv to nc

func MakePipe() Pipe {
	return Pipe{make(chan []float32), make(chan []float32)}
}

type Pipe struct {
	Data, Return chan []float32
}

func (p *Pipe) SendPipe() SendPipe {
	return SendPipe{p.Data, p.Return}
}

func (p *Pipe) RecvPipe() RecvPipe {
	return RecvPipe{p.Data, p.Return}
}

type SendPipe struct {
	Send       chan<- []float32 // Send slices here
	ReturnSend <-chan []float32 // Used slices go back here for recycling
}

type RecvPipe struct {
	Recv       <-chan []float32 // Receive slices here
	ReturnRecv chan<- []float32 // Send used slices back here for recycling
}
