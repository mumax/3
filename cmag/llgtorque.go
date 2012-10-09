package cmag

// #include "llgtorque.h"
import "C"

import (
	"nimble-cube/mag"
)

func LLGTorque(torque, m, B [3][]float32, alpha float32, bExt mag.Vector) {
	C.test()
}
