package unit

var (
	TIME   float64 = 1e-15 // seconds
	BFIELD float64 = 1     // Tesla
)

// Time unit in seconds.
func Time() float64 { return TIME }

// Magnetic induction unit in Tesla.
func BField() float64 { return BFIELD }
