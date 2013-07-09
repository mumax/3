package web

//
//import (
//	"code.google.com/p/mx3/cuda"
//	"code.google.com/p/mx3/data"
//	"code.google.com/p/mx3/engine"
//	"code.google.com/p/mx3/util"
//	"github.com/barnex/cuda5/cu"
//	"html/template"
//	"net/http"
//	"os"
//)
//
//var (
//	ui      = &guistate{}
//	uitempl = template.Must(template.New("gui").Parse(templText))
//)
//
//// http handler that serves main gui
//func gui(w http.ResponseWriter, r *http.Request) {
//	engine.InjectAndWait(func() { util.FatalErr(uitempl.Execute(w, ui)) })
//}
//
//type guistate struct{}
//
//func (s *guistate) Time() float32                    { return float32(engine.Time) }
//func (s *guistate) Mesh() *data.Mesh                 { return engine.Mesh() }
//func (s *guistate) Uname() string                    { return engine.UNAME }
//func (s *guistate) Version() string                  { return engine.VERSION }
//func (s *guistate) Pwd() string                      { pwd, _ := os.Getwd(); return pwd }
//func (s *guistate) Device() cu.Device                { return cu.CtxGetDevice() }
//func (s *guistate) Quants() map[string]engine.Getter { return engine.Quants }
//
//// world size in nm.
//func (s *guistate) WorldNm() [3]float64 {
//	w := engine.WorldSize()
//	return [3]float64{w[0] * 1e9, w[1] * 1e9, w[2] * 1e9}
//}
//
////const mib = 1024 * 2014
//// TODO: strangely this reports wrong numbers (x2 too low).
//// func (s *guistate) MemInfo() string { f, t := cu.MemGetInfo(); return fmt.Sprint(f/mib, "/", t/mib) }
//
//func (s *guistate) Solver() *cuda.Heun {
//	return &engine.Solver
//}
