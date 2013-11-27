package engine

import (
	"math"
)

// PMA domain wall tilt assuming straight wall.
var DWTiltPMA = NewGetScalar("ext_dwtilt", "rad", "PMA domain wall tilt", dwTiltPMA)

func dwTiltPMA() float64 {
	m := Download(&M)
	mz := m.Vectors()[Z][0] // slice0

	nx := Mesh().Size()[X]
	ny := Mesh().Size()[Y]
	// find domain wall at these y positions:
	y1 := 4
	y2 := ny - 5

	// search for x values where mz = 0 (=wall)
	x1, x2 := 0, 0
	for i := 1; i < nx; i++ {
		if mz[y1][i-1]*mz[y1][i] < 0 {
			x1 = i
		}
		if mz[y2][i-1]*mz[y2][i] < 0 {
			x2 = i
		}
	}
	angle := math.Atan(float64(x1-x2) / float64(y1-y2))
	return angle
}
