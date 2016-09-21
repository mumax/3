package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"reflect"
)

func init() {
	DeclFunc("AverageOf", AverageOf, "Evaluate quantity average")
}

type Q interface {
	NComp() int
	EvalTo(dst *data.Slice) // TODO: MSlice?
}

func MeshSize() [3]int {
	return Mesh().Size()
}

func SizeOf(q Q) [3]int {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh().Size()
	}
	// otherwise: default mesh
	return MeshSize()
}

func AverageOf(q Q) []float64 {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		average() []float64
	}); ok {
		return s.average()
	}
	// otherwise: default mesh
	buf := ValueOf(q)
	defer cuda.Recycle(buf)
	return sAverageMagnet(buf)
}

func NameOf(q Q) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Name() string
	}); ok {
		return s.Name()
	}
	return "unnamed." + reflect.TypeOf(q).String()
}

func UnitOf(q Q) string {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Unit() string
	}); ok {
		return s.Unit()
	}
	return "?"
}

func ValueOf(q Q) *data.Slice {
	// TODO: check for Buffered() implementation
	buf := cuda.Buffer(q.NComp(), SizeOf(q))
	q.EvalTo(buf)
	return buf
}

// dst += q
func AddTo(dst *data.Slice, q Q) {
	// TODO: check for AddTo implementation
	v := ValueOf(q)
	defer cuda.Recycle(v)
	cuda.Add(dst, dst, v)
}

// IsZero checks if q's multipliers are zero.
// Used to avoid launching cuda kernels that would not do any work.
//
// When returning true, q is definitely zero,
// when returning false, q may or may not be zero,
// and must be treated as if it's potentially non-zero.
func IsZero(q Q) bool {
	m := MSliceOf(q)
	for i := 0; i < m.NComp(); i++ {
		if m.Mul(i) != 0 {
			return false
		}
	}
	return true
}

func MSliceOf(q Q) cuda.MSlice {
	if q, ok := q.(interface {
		MSlice() cuda.MSlice
	}); ok {
		return q.MSlice()
	}
	return cuda.MakeMSlice(ValueOf(q), ones(q.NComp()))
}

var ones_ = [4]float64{1, 1, 1, 1}

func ones(n int) []float64 {
	return ones_[:n]
}

func EvalTo(q Q, dst *data.Slice) {
	util.AssertMsg(q.NComp() == dst.NComp() && SizeOf(q) == dst.Size(), "size mismatch")
	q.EvalTo(dst)
}

func EvalScript(q Q) (*data.Slice, bool) {
	buf := cuda.Buffer(q.NComp(), SizeOf(q))
	q.EvalTo(buf)
	return buf, true
}
