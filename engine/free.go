package engine

var onFree []func()

// TODO: mv to cuda/
func OnFree(f func()) {
	onFree = append(onFree, f)
}

func runOnFree() {
	for _, f := range onFree {
		f()
	}
	onFree = nil
}
