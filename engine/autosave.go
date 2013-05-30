package engine

import (
	"code.google.com/p/mx3/data"
	"fmt"
	"log"
)

// keeps info needed to decide when a quantity needs to be periodically saved
type autosave struct {
	period     float64 // How often to save
	start      float64 // Starting point
	count      int     // Number of times it has been autosaved
	autonum    int     // File number for output, may be > count when saved manually
	name, unit string
	// TODO: move these elsewhere:
	nComp int // number of components (scalar, vector, ...)
	mesh  *data.Mesh
}

func newAutosave(nComp int, name, unit string, m *data.Mesh) autosave {
	return autosave{nComp: nComp, name: name, unit: unit, mesh: m}
}

// Register a quantity for auto-saving every period (in seconds).
// Use zero period to disable auto-save.
func (a *autosave) Autosave(period float64) {
	log.Println("auto saving", a.name, "every", period, "s")
	a.period = period
	a.start = Time
	a.count = 0
}

// returns true when the time is right to save.
// after saving, saved() should be called
func (a *autosave) needSave() bool {
	if a.period == 0 {
		return false
	}
	t := Time - a.start
	return t-float64(a.count)*a.period >= a.period
}

// to be called after saving the quantity
func (a *autosave) saved() {
	a.count++
	a.autonum++
}

// Returns filename to save the quantity and increments the auto number.
func (a *autosave) autoFname() string {
	fname := fmt.Sprintf("%s/%s%06d.dump", OD, a.name, a.autonum)
	a.autonum++
	return fname
}

// NComp returns the number of components.
func (b *autosave) NComp() int { return b.nComp }
