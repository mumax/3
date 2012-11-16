package dump

//import (
//	"code.google.com/p/nimble-cube/core"
//	"code.google.com/p/nimble-cube/nimble"
//)
//
//// TODO: mv to nimble/
//type Autosaver struct {
//	out   *Writer
//	data  nimble.RChanN
//	every int
//}
//
//func RunAutosaver(fname string, data_ nimble.Chan, every int) *Autosaver {
//	r := new(Autosaver)
//	r.out = NewWriter(core.OpenFile(core.OD+fname), CRC_ENABLED)
//	data := data_.ChanN().NewReader()
//	r.out.Components = data.NComp()
//	r.out.MeshSize = data.Mesh().Size()
//	r.data = data
//	r.every = every
//	nimble.Stack(r)
//	return r
//}
//
//func (r *Autosaver) Run() {
//	N := r.data.Mesh().NCell()
//
//	for i := 0; ; i++ {
//		output := r.data.ReadNext(N) // TODO: could read comp by comp...
//		if i%r.every == 0 {
//			i = 0
//			core.Debug("dump")
//			r.out.WriteHeader()
//			for c := 0; c < r.data.NComp(); c++ {
//				r.out.WriteData(output[c].Host())
//			}
//			r.out.WriteHash()
//		}
//		r.data.ReadDone()
//	}
//}
