package core

type Quant struct {
	*Mesh
	list  [][]float32
	array [][][][]float32
	RWMutex
}

func NewQuant(mesh *Mesh, nComp int) *Quant {
	q := new(Quant)
	q.Mesh = mesh

	q.list = make([][]float32, nComp)
	q.array = make([][][][]float32, nComp)
	for c := range q.list {
		q.list[c] = make([]float32, q.NCell())
		q.array[c] = Reshape(q.list[c], q.GridSize())
	}

	return nil
}

// Number of components.
func (q *Quant) NComp() int { return len(q.list) }
