package mx

func Resample(in *Slice, N0, N1, N2 int) *Slice {
	m1 := in.Mesh()
	w := m1.WorldSize()
	n0, n1, n2 := float64(N0), float64(N1), float64(N2)
	pbc := m1.PBC()
	m2 := NewMesh(N0, N1, N2, w[0]/n0, w[1]/n1, w[2]/n2, pbc[:]...)
	out := NewSliceMemtype(in.NComp(), m2, in.MemType())

}
