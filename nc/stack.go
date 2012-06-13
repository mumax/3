package nc

type stack [][]float32

func (s *stack) push(slice []float32) {
	(*s) = append((*s), slice)
}

func (s *stack) pop() (slice []float32) {
	if len(*s) == 0 {
		return nil
	}
	slice = (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return
}

type gpuStack []GpuFloats

func (s *gpuStack) push(slice GpuFloats) {
	(*s) = append((*s), slice)
}

func (s *gpuStack) pop() (slice GpuFloats) {
	if len(*s) == 0 {
		return 0
	}
	slice = (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return
}
