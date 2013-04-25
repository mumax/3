package engine

import "code.google.com/p/mx3/data"

// Output handle
type Handle interface {
	// Register a quantity for auto-saving every period (in seconds).
	// Use zero period to disable auto-save.
	Autosave(period float64)
}

// Output handle that also support manual single-shot saving.
type Buffered interface {
	Handle                 // auto-save
	Save()                 // single-shot save with auto file name
	SaveAs(fname string)   // single-shot save with manual file name
	Download() *data.Slice // CPU-accessible slice
	Average() []float64
	MaxNorm() float64
}
