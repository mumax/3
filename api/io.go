package api

import (
	"nimble-cube/core"
)

// Sets the output directory where auto-saved files will be stored.
func SetOD(od string) {
	core.SetOD(od)
}

// Gets the output directory.
func GetOD() string {
	return core.OD
}

// Logs a message.
func Log(msg ...interface{}){
	core.Log(msg...)
}

// Retruns a new mesh with N0 x N1 x N2 cells of size cellx x celly x cellz.
// Optional periodic boundary conditions (pbc): number of repetitions
// in X, Y, Z direction. PBC 0, 0, 0 means no periodicity.
func NewMesh(N0, N1, N2 int, cellx, celly, cellz float64, pbc ...int) *core.Mesh {
	return core.NewMesh(N0, N1, N2, cellx, celly, cellz, pbc...)
}
