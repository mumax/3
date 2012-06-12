package nc

// Postman sends/receives data to/from arrays of channels.
// TODO: Recv*(): check ok, close outputs + goExit()+log if not

//****
//Need to debug data flow.
//IDEA:
//plumber.connect registers channels (not pointers, the actual channels)
//at bookkeeper. There will be chan float64 and chan []float32.
//no tag names required, just the master channels.
//use them to index fill map
//var fill64 map[chan float64]int
//user master channels for count (equality for <- and -> chan)
//fill--, fill++
//if debug: output to animation file, use runtime.Caller to identify caller
//who incs, decs the chan.
//there should be just one sender, one receiver.
//output to animation file

func SendFloat64(fanout []chan<- float64, value float64) {
	for _, ch := range fanout {
		ch <- value
	}
}

func Send(fanout []chan<- []float32, value []float32) {
	for _, ch := range fanout {
		ch <- value
	}
}

func Send3(vectorFanout [3][]chan<- []float32, value [3][]float32) {
	for comp := 0; comp < 3; comp++ {
		for _, ch := range vectorFanout[comp] {
			ch <- value[comp]
		}
	}
}

func Recv(Chan <-chan []float32) []float32 {
	if Chan == nil {
		panic("Recv on nil chan")
	}
	return <-Chan
}

func RecvFloat64(Chan <-chan float64) float64 {
	if Chan == nil {
		panic("RecvFloat64 on nil chan")
	}
	return <-Chan
}

func Recv3(vectorChan [3]<-chan []float32) [3][]float32 {
	if vectorChan[X] == nil {
		panic("Recv3 on nil chan")
	}
	return [3][]float32{<-vectorChan[X], <-vectorChan[Y], <-vectorChan[Z]}
}

// The postman always syncs twice.
