package safe

import "testing"

func TestFloat32sSlice(test *testing.T) {
	a := MakeFloat32s(100)
	defer a.Free()
}
