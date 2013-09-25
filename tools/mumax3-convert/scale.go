package main

//import (
//	"github.com/mumax/3/dump"
//	"github.com/mumax/3/core"
//)
//
//func rescale(f *dump.Frame, factor int) {
//	core.Assert(factor > 1)
//	for i:=range f.MeshSize{
//		size := f.MeshSize[i] / factor
//		if size == 0{size=1}
//		f.MeshStep[i] *= float64(f.MeshSize[i])/float64(size)
//		f.MeshSize[i] = size
//	}
//	in := f.Tensors()
//	out := core.MakeTensors(f.NComp(), f.MeshSize)
//	core.ScaleNearest(in, out)
//	f.Data = core.Contiguous4D(out)
//}
//
