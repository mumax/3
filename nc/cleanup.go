package nc

var atexit []func()

// Add a function to be executed at program exit
func AtExit(cleanup func()) {
	atexit = append(atexit, cleanup)
	Debug("atexit:", cleanup)
}

func CleanExit() {
	Log("exiting")
	for _, f := range atexit {
		Log("cleanup:", f)
		f()
	}
}
