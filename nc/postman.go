package nc

// Postman sends/receives data to/from arrays of channels.
// TODO: Recv*(): check ok, close outputs + goExit()+log if not

//****
//Need to debug data flow.
//IDEA:
//plumber.connect registers channels (not pointers, the actual channels)
//at bookkeeper. There will be chan float64 and chan Block.
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
	//if len(fanout)==0{panic("Send to nil")}
	for _, ch := range fanout {
		ch <- value
	}
}

// Send to channels with bookkeeping. 
// Should always be used instead of the <- operator.
func Send(fanout []chan<- Block, value Block) {
	// safety first
	if len(fanout) == 0 {
		panic("Send to nil")
	}

	// reference counting: Send adds len() copies, but removes one as sender looses a reference.
	count := int32(len(fanout)) - 1
	if count != 0 { // many are 0
		incr(value, count)
	}

	for _, ch := range fanout {
		ch <- value
	}
}

func SendGpu(fanout []chan<- GpuBlock, value GpuBlock) {
	if len(fanout) == 0 {
		panic("Send to nil")
	}

	count := int32(len(fanout)) - 1
	if count != 0 {
		incrGpu(value, count)
	}

	for _, ch := range fanout {
		ch <- value
	}
}

func Send3(fanout [3][]chan<- Block, value [3]Block) {
	if len(fanout[X]) == 0 {
		panic("Send to nil")
	}
	count := int32(len(fanout[X])) - 1
	if count != 0 {
		incr3(value, count)
	}
	for comp := 0; comp < 3; comp++ {
		for _, ch := range fanout[comp] {
			ch <- value[comp]
		}
	}
}

// Receive from channel with bookkeeping. 
// Should always be used instead of the <- operator.
func Recv(Chan <-chan Block) Block {
	if Chan == nil {
		panic("Recv on nil chan")
	}
	return <-Chan
}

func RecvGpu(Chan <-chan GpuBlock) GpuBlock {
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

func Recv3(vectorChan [3]<-chan Block) [3]Block {
	if vectorChan[X] == nil {
		panic("Recv3 on nil chan")
	}
	return [3]Block{<-vectorChan[X], <-vectorChan[Y], <-vectorChan[Z]}
}

// The postman always syncs twice.
