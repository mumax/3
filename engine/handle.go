package engine

// Output handle
type Handle interface {
	// Register a quantity for auto-saving every period (in seconds).
	// Use zero period to disable auto-save.
	Autosave(period float64)
}
