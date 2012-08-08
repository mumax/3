package nc

import (
	"reflect"
	"testing"
)

func TestGpuBlock(test *testing.T) {
	LockCudaThread()

	size := [3]int{2, 3, 4}
	h1 := MakeBlock(size)
	for i := range h1.List {
		h1.List[i] = float32(i)
	}

	d := MakeGpuBlock(size)
	d.CopyHtoD(h1)
	h2 := d.Host()

	if !reflect.DeepEqual(h1, h2) {
		test.Error(h1, "!=", h2)
	}
}
