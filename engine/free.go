package engine

var onFree []func()

func OnFree(f func()) {
	onFree = append(onFree, f)
}

func runOnFree() {
	for _, f := range onFree {
		f()
	}
	onFree = nil
}
