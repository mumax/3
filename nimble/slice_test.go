package nimble

import "testing"

func TestSlice(t *testing.T) {
	a := make([]float32, 100)
	s := Float32ToSlice(a)
	if len(a) != s.Len() {
		t.Fail()
	}

	a = a[10:42]
	s = s.Slice(10, 42)
	if len(a) != s.Len() {
		t.Fail()
	}

	b := s.Host()
	if &b[0] != &a[0] {
		t.Fail()
	}
}
