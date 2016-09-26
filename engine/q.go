package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"reflect"
)

type Q interface {
	NComp() int
	EvalTo(dst *data.Slice)
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

func MeshOf(q Q) *data.Mesh {
	// quantity defines its own, custom, implementation:
	if s, ok := q.(interface {
		Mesh() *data.Mesh
	}); ok {
		return s.Mesh()
	}
	return Mesh()
}

func ValueOf(q Q) *data.Slice {
	// TODO: check for Buffered() implementation
	buf := cuda.Buffer(q.NComp(), SizeOf(q))
	q.EvalTo(buf)
	return buf
}

// Temporary shim to fit Slice into EvalTo
func EvalTo(q interface {
	Slice() (*data.Slice, bool)
}, dst *data.Slice) {
	v, r := q.Slice()
	if r {
		defer cuda.Recycle(v)
	}
	data.Copy(dst, v)
}

//func EvalTo(q Q, dst *data.Slice) {
//	util.AssertMsg(q.NComp() == dst.NComp() && SizeOf(q) == dst.Size(), "size mismatch")
//	q.EvalTo(dst)
//}

func EvalScript(q Q) (*data.Slice, bool) {
	buf := cuda.Buffer(q.NComp(), SizeOf(q))
	q.EvalTo(buf)
	return buf, true
}
