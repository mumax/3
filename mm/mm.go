package mm

import (
	"log"
	. "nimble-cube/nc"
)

var (
	size   [3]int
	m      VectorBlock // reduced magnetization
	Heff   VectorBlock // effective field
	gamma  AnyScalarBlock
	alpha  AnyScalarBlock
	torque VectorBlock
)

func Main() {
	log.Println("nimble-cube/mm")

	N0, N1, N2 := 1, 4, 8
	size = [3]int{N0, N1, N2}

	m = MakeVectorBlock(size)
	Heff = MakeVectorBlock(size)

	m[X].Memset(1)
	m[Y].Memset(0)
	m[Z].Memset(0)

	a := MakeBlock(size)
	a.Memset(0.01)
	alpha = a
}

func UpdateTorque() {

}
