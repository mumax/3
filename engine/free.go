package engine

var freelist []interface {
	Free()
}

func FreeAll() {
	for _, f := range freelist {
		f.Free()
	}
	freelist = nil
}

func RegisterFree(f interface {
	Free()
}) {
	freelist = append(freelist, f)
}
