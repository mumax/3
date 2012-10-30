package core

type Slice struct {
	list  []float32
	array [][][]float32
}

func (s *Slice) Slice(a, b int) Slice {
	return Slice{s.list[a:b], nil}
}
