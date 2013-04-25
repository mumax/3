package engine

var onfree []func()

// TODO: mv to cuda/
func onFree(f func()) {
	onfree = append(onfree, f)
}

func runOnFree() {
	for _, f := range onfree {
		f()
	}
	onfree = nil
}
