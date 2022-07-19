package cuda

import (
	"testing"
	"unsafe"

	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/util"
)

// test input data
var in1, in2, in3 *data.Slice

func initTest() {
	if in1 != nil {
		return
	}
	{
		inh1 := make([]float32, 1000)
		for i := range inh1 {
			inh1[i] = float32(i)
		}
		in1 = toGPU(inh1)
	}
	{
		inh2 := make([]float32, 100000)
		for i := range inh2 {
			inh2[i] = -float32(i) / 100
		}
		in2 = toGPU(inh2)
	}
}

func toGPU(list []float32) *data.Slice {
	mesh := [3]int{1, 1, len(list)}
	h := sliceFromList([][]float32{list}, mesh)
	d := NewSlice(1, mesh)
	data.Copy(d, h)
	return d
}

func TestReduceSum(t *testing.T) {
	initTest()
	result := Sum(in1)
	if result != 499500 {
		t.Error("got:", result)
	}
}

func TestReduceDot(t *testing.T) {
	initTest()

	// test for 1 comp
	a := toGPU([]float32{1, 2, 3, 4, 5})
	b := toGPU([]float32{5, 4, 3, -1, 2})
	result := Dot(a, b)
	if result != 5+8+9-4+10 {
		t.Error("got:", result)
	}

	// test for 3 comp
	const N = 32
	mesh := [3]int{1, 1, N}
	c := NewSlice(3, mesh)
	d := NewSlice(3, mesh)
	Memset(c, 1, 2, 3)
	Memset(d, 4, 5, 6)
	result = Dot(c, d)
	if result != N*(4+10+18) {
		t.Error("got:", result)
	}
}

func TestReduceMaxAbs(t *testing.T) {
	result := MaxAbs(in1)
	if result != 999 {
		t.Error("got:", result)
	}
	result = MaxAbs(in2)
	if result != 999.99 {
		t.Error("got:", result)
	}
}

func sliceFromList(arr [][]float32, size [3]int) *data.Slice {
	ptrs := make([]unsafe.Pointer, len(arr))
	for i := range ptrs {
		util.Argument(len(arr[i]) == prod(size))
		ptrs[i] = unsafe.Pointer(&arr[i][0])
	}
	return data.SliceFromPtrs(size, data.CPUMemory, ptrs)
}
