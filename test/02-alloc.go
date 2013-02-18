// +build ignore

package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/data"
)

func main() {
	cuda.Init()

	N0, N1, N2 := 2, 16, 32
	c0, c1, c2 := 1e-9, 1e-9, 1e-9
	mesh := data.NewMesh(N0, N1, N2, c0, c1, c2)

	m1 := cuda.NewQuant(3, mesh).Data()
	defer m1.Free()
	m1.HostCopy().Host()

	m2 := cuda.NewUnifiedSlice(3, mesh)
	defer m2.Free()
	m2.Host()
	m2.HostCopy().Host()

	m3 := data.NewSlice(3, mesh)
	m3.Host()
	//m3.HostCopy()

	//	data.Copy(m3, m2)
	//	data.Copy(m2, m3)
	//	data.Copy(m1, m2)
	//	data.Copy(m2, m1)
	//	data.Copy(m3, m1)
	//	data.Copy(m1, m3)
}
