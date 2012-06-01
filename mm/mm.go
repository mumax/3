package mm

import (
	"log"
	//"nimble-cube/nc"
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

	initSize()
	go RunGC()

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


CONCEPT:

go RunTorque()

func RunTorque(){
	// replace by := notation
	var recvm chan<- float32[] = recv(m) // engine inserts tee if needed 
		// engine uses runtime.Caller to construct (purely informative) dependency graph:  torque <- RunTorque <- (m, h)
	var recvh chan<- float32[] = recv(h)
	var sendtorque <-chan[]float32 = send(torque)

	for{
		buf := <- getbuffer
		m:=<-recvm
		h:=<-recvh
		torque(buf, m, h)
		sendtorque <- torque
		recycle <- m
		recycle <- h
	}
}
