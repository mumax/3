package mm

import (
	. "nimble-cube/nc"
	"log"
)

var (
	size [3]int
	m    VectorBlock // reduced magnetization
	Heff    VectorBlock // effective field
)

func Main(){
	log.Println("nimble-cube/mm")

	N0, N1, N2 := 1, 128, 1024
	size = [3]int{N0, N1, N2}

	m = MakeVectorBlock(size)
	Heff = MakeVectorBlock(size)

	
}
