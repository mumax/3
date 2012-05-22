package mx

// Buffers a small []float32, growing it as necessary.
type Buffer []float32

// Return a buffer of size n, valid until the next call.
// Re-uses the previous buffer if possible.
func (b *Buffer) MakeBuffer(n int) {
	if len(*b) == n {
		return
	}
	if cap(*b) >= n {
		*b = (*b)[:n]
	}
	*b = make([]float32, n)
}

func (b *Buffer) Reset() {
	*b = (*b)[0:1]
}
