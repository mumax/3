package nc

import "sync"

type stack struct {
	array []Block
	sync.Mutex
}

// thread-safe
func (s *stack) push(slice Block) {
	s.Lock()
	s.array = append(s.array, slice)
	s.Unlock()
}

// thread-safe
func (s *stack) pop() (slice Block) {
	s.Lock()
	if len(s.array) != 0 {
		slice = (s.array)[len(s.array)-1]
		s.array = s.array[:len(s.array)-1]
	}
	s.Unlock()
	return
}

type gpuStack struct {
	array []GpuBlock
	sync.Mutex
}

// thread-safe
func (s *gpuStack) push(slice GpuBlock) {
	s.Lock()
	s.array = append(s.array, slice)
	s.Unlock()
}

// thread-safe
func (s *gpuStack) pop() (slice GpuBlock) {
	s.Lock()
	if len(s.array) != 0 {
		slice = s.array[len(s.array)-1]
		s.array = s.array[:len(s.array)-1]
	}
	s.Unlock()
	return
}
