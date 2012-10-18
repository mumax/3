package core

//
//type Slice interface {
//}
//
//type Quant struct {
//	tag   string
//	unit  string
//	mesh  *Mesh
//	data  Slice
//	mutex []*RWMutex
//}
//
//func NewQuant(nComp int, m *Mesh, tag string, unit string) *Quant {
//	q := &Quant{mesh: m, tag: tag, unit: unit}
//	q.data = MakeTensor(nComp, m.Size())
//	q.mutex = make([]*RWMutex, nComp)
//	N := Prod(m.Size())
//	for i := range q.mutex {
//		q.mutex[i] = NewRWMutex(N, tag) // TODO: rm tag
//	}
//	return q
//}
//
//func(q*Quant)Writer()*Writer{
//
//}
//
//type Writer struct{
//
//}
//
//type Tensor struct {
//	list [][]float32
//}
//
//func MakeTensor(nComp int, size [3]int) *Tensor {
//	list := make([][]float32, nComp)
//	n := Prod(size)
//	for i := range list {
//		list[i] = make([]float32, n)
//	}
//	return &Tensor{list}
//}
