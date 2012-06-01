package mm

import (
	"log"
)

type DoubleBuffer struct {
	RecvPipe
	SendPipe
	InBuf, OutBuf   []float32
	nextIn, nextOut int
	swap            chan []float32
}

func MakeDoubleBuffer(size int) DoubleBuffer {
	var b DoubleBuffer
	b.InBuf = make([]float32, size)
	b.OutBuf = make([]float32, size)
	return b
}

func (b *DoubleBuffer) Size() int { return len(b.InBuf) }

func (b *DoubleBuffer) Cycle() {
	go b.cycleIn()
	b.cycleOut()
}

func (b *DoubleBuffer) cycleIn() {
	size := b.Size()
	in := 0

	for in < size {
		recv := <-b.Recv
		copy(b.InBuf[in:], recv)
		in += len(recv)
		b.ReturnRecv <- recv
	}

	log.Println("DoubleBuffer Recv cycle complete")

}

func (b *DoubleBuffer) cycleOut() {
	size := b.Size()
	out := 0
	backin := 0

	for backin < size {
		var send chan<-[]float32
		var sendme []float32
		if out < size {
			send = b.Send
			sendme = b.OutBuf[out : out+warp]
		}
		select {
		case send <- sendme:
			out += warp
		case garbage := <-b.ReturnSend:
			backin += len(garbage)
		}
	}

	log.Println("DoubleBuffer Send cycle complete")

}
