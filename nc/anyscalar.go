package nc

type AnyScalar interface {
	NScalar() int
	Range(i1, i2 int) []float32
}
