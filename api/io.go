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
