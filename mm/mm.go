package mm

import (
	. "nimble-cube/nc"
	"os"
)

var (
	size   [3]int
	m      VectorBlock // reduced magnetization
	Heff   VectorBlock // effective field
	gamma  AnyScalar
	alpha  AnyScalar
	torque VectorBlock
)

func Main() {
	PrintInfo(os.Stdout)

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

	Println(alpha)

}

func UpdateTorque() {

}
