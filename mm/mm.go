package mm

import (
	"log"
	"nimble-cube/nc"
	"runtime"
)

var (
	size  [3]int // 3D geom size
	N     int    // product of size
	warp  int    // buffer size for Range()
	Debug = false

	mChan     VecChan
	alphaChan Chan
	hChan     VecChan
)

func Main() {

	initSize()
	go RunGC()

	mChan = MakeVecChan(N / warp)
	alphaChan = MakeChan(0)
	hChan = MakeVecChan(0)

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

	const dt = 0.001
	const Nsteps = 10

	// seed m
	mx := make([]float32, warp)
	my := make([]float32, warp)
	mz := make([]float32, warp)
	nc.Memset(mx, 1)
	mInit := [3][]float32{mx, my, mz}
	mOffline := MakeVecChan(N / warp)
	for I := 0; I < N; I += warp {
		mOffline.Send(mInit) // [I]
	}

	RunSolver(mOffline, Nsteps, dt)

	for I := 0; I < N; I += warp {
		log.Println(mOffline.Recv())
	}

}

func RunSolver(mOffline VecChan, Nsteps int, dt float32) {

	for I := 0; I < N; I += warp {
		mChan.Send(mOffline.Recv())
	}

	for step := 0; step < Nsteps; step++ {

		// loop over blocks
		for I := 0; I < N; I += warp {

			newMList := VecBuffer()
			//alphaList := alphaChan.Recv()
			mList := mChan.Recv()
			hList := hChan.Recv()

			// loop inside blocks
			for i := range newMList[X] {
				var m nc.Vector
				var h nc.Vector
				var newM nc.Vector
				m[X], m[Y], m[Z] = mList[X][i], mList[Y][i], mList[Z][i]
				h[X], h[Y], h[Z] = hList[X][i], hList[Y][i], hList[Z][i]
				//alpha := alphaList[i]

				mxh := m.Cross(h)
				tq := mxh //.Sub(m.Cross(mxh).Scale(alpha))
				newM = m.MAdd(dt, tq)

				newMList[X][i] = newM[X]
				newMList[Y][i] = newM[Y]
				newMList[Z][i] = newM[Z]
			}
			if step != Nsteps-1 {
				mChan.Send(newMList)
			} else {
				mOffline.Send(newMList)
			}

		}
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
