package unit

var (
	Time   float64 = 1e-15 // seconds
	B float64 = 1     // Tesla
	Length float64 = 1e-9  // meter
)

// Time unit in seconds.
func Time() float64 { return TIME }

// Magnetic induction unit in Tesla.
func BField() float64 { return BFIELD }

// Length unit in meters.
func Length() float64 { return LENGTH }
