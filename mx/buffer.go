package mx

type Buffer32 struct {
	buf []float32
}

func (b *Buffer32) Buffer(n int) []float32 {
	if len(b.buf) == n {
		return b.buf
	}
	if cap(b.buf) >= n {
		return b.buf[:n]
	}
	b.buf = make([]float32, n)
	return b.buf
}
