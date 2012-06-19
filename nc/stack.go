package nc

type stack []Block

func (s *stack) push(slice Block) {
	(*s) = append((*s), slice)
}

func (s *stack) pop() (slice Block) {
	if len(*s) == 0 {
		return // nil
	}
	slice = (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return
}

type gpuStack []GpuBlock

func (s *gpuStack) push(slice GpuBlock) {
	(*s) = append((*s), slice)
}

func (s *gpuStack) pop() (slice GpuBlock) {
	if len(*s) == 0 {
		return
	}
	slice = (*s)[len(*s)-1]
	*s = (*s)[:len(*s)-1]
	return
}
