package engine

type Free interface {
	Free()
}

var freelist []Free

func FreeAll() {
	for _, f := range freelist {
		f.Free()
	}
	freelist = nil
}

func RegisterFree(f Free) {
	freelist = append(freelist, f)
}
