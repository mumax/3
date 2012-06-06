package mm

import (
	"log"
	//. "nimble-cube/nc"
)

var probeNames = make(map[string]int) // Set of used probe names

type Probe3Box struct {
	data chan [3][]float32
	name string
}

//func Probe3(stream *FanIn3, name string) {
//	if _, ok := probeNames[name]; ok {
//		panic("Probe3: name already in use: " + name)
//	}
//	probeNames[name] = 1
//	b := Probe3Box{stream.FanOut(DefaultBufSize()), name}
//	go b.Run()
//}

func (b *Probe3Box) Run() {
	for {
		slice := <-b.data
		log.Println(b.name, slice)
		// RECYLCE
	}
}

//func RunTorque(tick <-chan float32, torque VecChan, mRecv VecRecv, hRecv VecRecv, alphaRecv Recv) {
//
//	// loop over blocks
//	for I := 0; I < N; I += warp {
//
//		torqueSlice := VecBuffer()
//		//alphaList := alphaChan.Recv()
//		mList := mRecv.Recv()
//		hList := hRecv.Recv()
//		_ = alphaRecv.Recv()
//
//		// loop inside blocks
//		for i := range torqueSlice[X] {
//			var m Vector
//			var h Vector
//			m[X], m[Y], m[Z] = mList[X][i], mList[Y][i], mList[Z][i]
//			h[X], h[Y], h[Z] = hList[X][i], hList[Y][i], hList[Z][i]
//			//alpha := alphaList[i]
//
//			mxh := m.Cross(h)
//			tq := mxh //.Sub(m.Cross(mxh).Scale(alpha))
//
//			torqueSlice[X][i] = tq[X]
//			torqueSlice[Y][i] = tq[Y]
//			torqueSlice[Z][i] = tq[Z]
//		}
//		torqueChan.Send(torqueSlice)
//	}
//}
