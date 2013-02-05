package main

import (
	"fmt"
	"nimble-cube/dump"
	"nimble-cube/gpu"
	"nimble-cube/gpu/conv"
	"nimble-cube/mag"
	. "nimble-cube/nimble"
	"os"
)

func main() {
	SetOD("uni4.out")
	InitGraph(OD + "graph.dot")

	N0, N1, N2 := 1, 32, 128
	cx, cy, cz := 3e-9, 3.125e-9, 3.125e-9
	mesh := NewMesh(N0, N1, N2, cx, cy, cz)

	m := MakeChan3("m", "", mesh)
	hd := MakeChan3("Hd", "", mesh)

	acc := 8
	kernel := mag.BruteKernel(ZeroPad(mesh), acc)
	Stack(conv.NewSymmetricHtoD(mesh, kernel, m.NewReader(), hd))

	Msat := 1.0053
	aex := mag.Mu0 * 13e-12 / Msat
	hex := MakeChan3("Hex", "", mesh)
	Stack(mag.NewExchange6(m.NewReader(), hex, mesh, aex))

	heff := MakeChan3("Heff", "", mesh)
	Stack(NewAdder3(heff, hd.NewReader(), hex.NewReader()))

	Gheff := gpu.RunUploader("HeffGPU", heff).Chan3()
	Gm := gpu.RunUploader("Gm", m).Chan3()

	const alpha = 1
	Gtorque := gpu.RunLLGTorque("Gtorque", Gm, Gheff, alpha).Output()
	torque := gpu.RunDownloader("torque", Gtorque).Chan3()

	const dt = 50e-15
	solver := mag.NewEuler(m, torque.NewReader(), mag.Gamma0, dt)
	mag.SetAll(m.UnsafeArray(), mag.Uniform(0, 0.1, 1))

	Stack(dump.NewAutosaver("h.dump", hd.NewReader(), 10))
	Stack(dump.NewAutosaver("m.dump", m.NewReader(), 10))
	Stack(dump.NewAutosaver("hex.dump", hex.NewReader(), 10))
	Stack(dump.NewAutotable("m.table", m.NewReader(), 10))

	RunStack()

	solver.Steps(100)
	res := m.UnsafeArray()
	got := [3]float32{res[0][0][0][0], res[1][0][0][0], res[2][0][0][0]}
	expect := [3]float32{-0.029639997, 0.16420172, 0.98598135}
	Log("result:", got)
	if got != expect {
		Fatal(fmt.Errorf("expected: %v", expect))
	} else {
		Log("OK")
	}

	ProfDump(os.Stdout)
	Cleanup()
}
