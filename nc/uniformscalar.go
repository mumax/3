package nc

import (
	"fmt"
)

type UniformScalar struct {
	value float32
	buf   []float32
}

func NewUniformScalar() *UniformScalar { return &UniformScalar{0, []float32{0}} }

func (s *UniformScalar) SetValue(value float32) {
	s.value = value
}

func (s *UniformScalar) Range(i1, i2 int) []float32 {
	lenHave := len(s.buf)
	lenWant := i2 - i1
	if lenWant == 0 {
		return []float32{}
	}
	if lenHave == lenWant {
		if s.buf[0] != s.value {
			Memset(s.buf, s.value)
		}
		return s.buf
	}
	s.buf = ResizeBuffer(s.buf, lenWant)
	Memset(s.buf, s.value)
	return s.buf
}

func (s *UniformScalar) String() string { return fmt.Sprint(s.value) }

func Memset(a []float32, value float32) {
	for i := range a {
		a[i] = value
	}
}

func ResizeBuffer(buf []float32, lenWant int) []float32 {
	lenHave := len(buf)
	switch {
	case lenHave == lenWant:
		return buf
	case lenHave > lenWant:
		return buf[:lenWant]
	case cap(buf) >= lenWant:
		newBuf := buf[:lenWant]
		return newBuf
	default:
		return make([]float32, lenWant)
	}
	return nil
}
