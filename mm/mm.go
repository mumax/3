package mm

import (
	"fmt"
	"log"
	"nimble-cube/nc"
)

var (
	size [3]int // 3D geom size
	N    int    // product of size
	warp int    // buffer size for Range()

	//m0, m1 VectorBlock // reduced magnetization, at old and new t
	//heff   VectorBlock // effective field
	//gamma  float32     // gyromagnetic ratio
	//alpha  float32     // damping coefficient
	//torque VectorBlock // dm/dt
)

func Main() {

	log.Println("1058")
	initSize()
	go RunGC()

	mChan := MakeVecChan(0)
	alphaChan := MakeChan(0)
	hChan := MakeVecChan(0)
	torqueChan := MakeVecChan(0)

	// send const M
	go func() {
		var m [3][]float32
		m[X] = make([]float32, warp)
		m[Y] = make([]float32, warp)
		m[Z] = make([]float32, warp)
		nc.Memset(m[X], 1)

		for {
			mChan.Send(m)
		}
	}()

	// send const H
	go func() {
		var h [3][]float32
		h[X] = make([]float32, warp)
		h[Y] = make([]float32, warp)
		h[Z] = make([]float32, warp)
		nc.Memset(h[Y], 1)

		for {
			hChan.Send(h)
		}
	}()

	// send const alpha
	go func() {
		alpha := make([]float32, warp)
		nc.Memset(alpha, 0.05)

		for {
			alphaChan.Send(alpha)
		}
	}()

	// run torque
	go func() {
		for {
			torque := VecBuffer()
			//alphaList := alphaChan.Recv()
			mList := mChan.Recv()
			hList := hChan.Recv()

			for i := range torque[X] {
				var m nc.Vector
				var h nc.Vector
				m[X], m[Y], m[Z] = mList[X][i], mList[Y][i], mList[Z][i]
				h[X], h[Y], h[Z] = hList[X][i], hList[Y][i], hList[Z][i]
				//alpha := alphaList[i]

				mxh := m.Cross(h)
				tq := mxh //.Sub(m.Cross(mxh).Scale(alpha))
				torque[X][i] = tq[X]
				torque[Y][i] = tq[Y]
				torque[Z][i] = tq[Z]
			}

			torqueChan.Send(torque)
		}
	}()

	for I := 0; I < N; I += warp {
		fmt.Println("torque:", torqueChan.Recv())
	}

}

func initSize() {
	N0, N1, N2 := 1, 4, 8
	size := [3]int{N0, N1, N2}
	N = N0 * N1 * N2

	log.Println("size:", size)
	N := N0 * N1 * N2
	log.Println("N:", N)

	// Find some nice warp size
	warp = 8
	for N%warp != 0 {
		warp--
	}
	log.Println("warp:", warp)
}

//CONCEPT:

//func RunTorque(){
//	// replace by := notation
//	var recvm chan<- float32[] = recv(m) // engine inserts tee if needed 
//		// engine uses runtime.Caller to construct (purely informative) dependency graph:  torque <- RunTorque <- (m, h)
//	var recvh chan<- float32[] = recv(h)
//	var sendtorque <-chan[]float32 = send(torque)
//
//	for{
//		buf := <- getbuffer
//		m:=<-recvm
//		h:=<-recvh
//		torque(buf, m, h)
//		sendtorque <- torque
//		recycle <- m
//		recycle <- h
//	}
//}
