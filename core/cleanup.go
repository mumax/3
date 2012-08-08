package core

var atexit []func()

// Add a function to be executed at program exit.
func AtExit(cleanup func()) {
	atexit = append(atexit, cleanup)
	Debug("atexit:", cleanup)
}

// Runs all functions stacked by AtExit().
func Cleanup() {
	Log("Cleanup")
	for _, f := range atexit {
		f()
	}
}
