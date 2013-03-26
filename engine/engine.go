package engine

//import (
//	"code.google.com/p/mx3/cuda"
//	"code.google.com/p/mx3/data"
//	"code.google.com/p/mx3/mag"
//	"code.google.com/p/mx3/util"
//	"log"
//	"sync"
//)
//
//// User inputs
//var (
//	Aex   ScalFn
//	Msat  ScalFn
//	Alpha ScalFn
//	Bext  VecFn = ConstVector(0, 0, 0)
//	DMI   VecFn = ConstVector(0, 0, 0)
//	Time  float64
//)
//
//var (
//	mesh     *data.Mesh
//	Solver   *cuda.Heun
//	m        *S
//	buffer   *S // holds H_effective or torque
//	_outbuf  *S
//	_hostbuf *S
//	vol      *data.Slice
//	demag    addFn
//	exch     addFn
//)
//
//var (
//	M, B_demag, B_exch, B_eff, Torque *Handle
//)
//
//type S struct {
//	_slice *data.Slice
//	_lock  sync.RWMutex
//}
//
//func newS() *S {
//	s := new(S)
//	s._slice = cuda.NewSlice(3, mesh)
//	return s
//}
//
//func (s *S) RLock() *data.Slice {
//	s._lock.RLock()
//	return s._slice
//}
//
//func (s *S) WLock() *data.Slice {
//	s._lock.Lock()
//	return s._slice
//}
//
//func (s *S) WUnlock() { s._lock.Unlock() }
//func (s *S) RUnlock() { s._lock.RUnlock() }
//
//func outBuf() *S {
//	if _outbuf == nil {
//		_outbuf = newS()
//	}
//	return _outbuf
//}
//
//type Handle struct {
//	autosave
//}
//
//func (h *Handle) Need() bool {
//	return h.needSave()
//}
//
//func (h *Handle) Send(s *data.Slice) {
//	log.Println("need save:", h.Need())
//	log.Println("\nsave", h.name)
//
//	//saveAndRecycle(s , h.fname(), Time)
//
//	h.saved()
//}
//
//func newHandle(name string) *Handle {
//	var h Handle
//	h.name = name
//	return &h
//}
//
//// Evaluates all quantities, possibly saving them in the meanwhile.
//func Eval() *data.Slice { // todo: output bool
//	//doOutput := Solver.GoodStep
//
//	M.output(m)
//
//	//	buf := buffer.Lock()
//	//	cuda.Memset(buf, 0, 0, 0)         // Need this in case demag is output, then we really add to.
//	//	addAndOutput(buf, demag, B_demag) // Does not add but sets, so it should be first.
//	//
//	//	addAndOutput(buffer, exch, B_exch)
//	//
//	//	bext := Bext()
//	//	cuda.AddConst(buffer.Slice, float32(bext[Z]), float32(bext[Y]), float32(bext[X]))
//	//	output(buffer, B_eff)
//	//
//	//
//	//	cuda.LLGTorque(buffer.Slice, m.Lock(), buffer.Slice, float32(Alpha()))
//	//	m.Unlock()
//	//	output(buffer, Torque)
//
//	out := buffer.RLock()
//	buffer.RUnlock()
//	return out
//}
//
//func (h *Handle) output(s *S) {
//	if h.Need() {
//		s.RLock()
//		s.RUnlock()
//	}
//}
//
//type addFn func(dst *data.Slice) // calculates quantity and add result to dst
//
//func addAndOutput(dst *S, addTo addFn, h *Handle) {
//	if h.Need() {
//		gb := outBuf().WLock()
//		addTo(buffer)
//		go func() {
//
//		}()
//		outBuf().WUnlock()
//
//		cuda.Madd2(dst, dst, buffer, 1, 1)
//		output(buffer, h)
//	} else {
//		addTo(dst)
//	}
//}
//
//func initialize() {
//	m = cuda.NewSlice(3, mesh)
//	M = newHandle("m")
//
//	mx, my, mz = m.Comp(0), m.Comp(1), m.Comp(2)
//	buffer = cuda.NewSlice(3, mesh)
//	vol = data.NilSlice(1, mesh)
//	Solver = cuda.NewHeun(m, Eval, 1e-15, Gamma0, &Time)
//
//	demag_ := cuda.NewDemag(mesh)
//	demag = func(dst *data.Slice) {
//		demag_.Exec(dst, m, vol, Mu0*Msat())
//	}
//	B_demag = newHandle("B_demag")
//
//	exch = func(dst *data.Slice) {
//		cuda.AddExchange(dst, m, Aex(), Mu0*Msat())
//	}
//	B_exch = newHandle("B_exch")
//
//	B_eff = newHandle("B_eff")
//
//	Torque = newHandle("torque")
//}
//
//func checkInited() {
//	if mesh == nil {
//		log.Fatal("need to set mesh first")
//	}
//}
//
//func SetMesh(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64) {
//	if mesh != nil {
//		log.Fatal("mesh already set")
//	}
//	mesh = data.NewMesh(Nz, Ny, Nx, cellSizeZ, cellSizeY, cellSizeX)
//	log.Println("set mesh:", mesh)
//	initialize()
//}
//
//
