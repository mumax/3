package mm

import (
	. "nimble-cube/nc"
	"os"
)

var magnet struct{
	size   [3]int // 3D geom size
	warp int // buffer size for Range()
	m      VectorBlock // reduced magnetization
	Heff   VectorBlock // effective field
	gamma  AnyScalar
	alpha  AnyScalar
	torque VectorBlock
}


func Main() {
	PrintInfo(os.Stdout)

	N0, N1, N2 := 1, 4, 8
	size := [3]int{N0, N1, N2}
	magnet.size = size
	Println("size:", magnet.size)
	N := N0 * N1 * N2
	Println("N:", N)

	// Find some nice warp size
	warp := 1024
	for N%warp != 0{
		warp--
	}
	magnet.warp = warp
	Println("warp:", magnet.warp)

	magnet.m = MakeVectorBlock(size)
	magnet.Heff = MakeVectorBlock(size)

	magnet.m[X].Memset(1)
	magnet.m[Y].Memset(0)
	magnet.m[Z].Memset(0)

	a := MakeBlock(size)
	a.Memset(0.01)
	magnet.alpha = a

	Println(magnet.alpha)

	a2 := NewUniformScalar()
	a2.SetValue(0.02)
	magnet.alpha = a2
	Println(magnet.alpha)

	UpdateTorque()
	Println(magnet.torque)
}

func UpdateTorque() {
	warp := magnet.warp
	N := magnet.torque.NVector()
	Torque := magnet.torque.Contiguous()
	M := magnet.m.Contiguous()
	for I := 0; I < N; I+= warp{
		torque := Torque.Range(I, I+warp)
		m := M.Range(I, I+warp)
		for i:=0; i<warp; i++{
			torque[i] = m[i]
		}
	}
}



