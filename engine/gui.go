package engine

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/gui"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"
)

var (
	quants    = make(map[string]Getter)
	params    = make(map[string]Param)
	KeepAlive = func() time.Time { return time.Time{} } // overwritten by gui server
	renderQ   = "m"                                     // quantity to display
)

type Param interface {
	NComp() int
	Unit() string
	GetVec() []float64
	setUniform(...float64)
}

type guidata struct {
	Quants map[string]Getter
	Params map[string]Param
}

// list of text box id's for component text boxes.
func (d *guidata) CompBoxIds(param string) []string {
	var e []string
	p := params[param]
	for i := 0; i < p.NComp(); i++ {
		e = append(e, fmt.Sprint("param_", param, i))
	}
	return e
}

// Start web gui on given port, blocks.
func Serve(port string) {
	log.Println("gui waiting for engine init")
	<-inited
	log.Println("engine inited, gui starting")

	data := &guidata{Quants: quants, Params: params}
	gui := gui.NewDoc(templText, data)
	KeepAlive = gui.KeepAlive

	http.Handle("/", gui)
	http.HandleFunc("/render/", serveRender)

	// geometry
	size := Mesh().Size()
	gui.SetValue("nx", size[2])
	gui.SetValue("ny", size[1])
	gui.SetValue("nz", size[0])
	cellSize := Mesh().CellSize()
	gui.SetValue("cx", cellSize[2]*1e9) // in nm
	gui.SetValue("cy", cellSize[1]*1e9)
	gui.SetValue("cz", cellSize[0]*1e9)
	gui.SetValue("wx", float64(size[2])*cellSize[2]*1e9)
	gui.SetValue("wy", float64(size[1])*cellSize[1]*1e9)
	gui.SetValue("wz", float64(size[0])*cellSize[0]*1e9)

	// solver
	gui.OnEvent("break", Pause)
	gui.OnEvent("run", inj(func() { Run(gui.Value("runtime").(float64)) }))
	gui.OnEvent("steps", inj(func() { Steps(gui.Value("runsteps").(int)) }))
	gui.OnEvent("fixdt", inj(func() { Solver.FixDt = gui.Value("fixdt").(float64) }))
	gui.OnEvent("mindt", inj(func() { Solver.MinDt = gui.Value("mindt").(float64) }))
	gui.OnEvent("maxdt", inj(func() { Solver.MaxDt = gui.Value("maxdt").(float64) }))
	gui.OnEvent("maxerr", inj(func() { Solver.MaxErr = gui.Value("maxerr").(float64) }))
	gui.OnEvent("sel_render", func() { renderQ = gui.Value("sel_render").(string) })

	// display
	gui.SetValue("sel_render", renderQ)

	// parameters
	for n, p := range params {
		n := n // closure caveats...
		p := p

		compIds := ((*guidata)(nil)).CompBoxIds(n)
		handler := func() {
			v := make([]float64, len(compIds))
			for comp, id := range compIds {
				v[comp] = gui.Value(id).(float64)
			}
			Inject <- func() { p.setUniform(v...) }
		}

		for _, id := range compIds {
			gui.OnEvent(id, handler)
		}

	}

	// process
	gui.SetValue("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))
	hostname, _ := os.Hostname()
	gui.SetValue("hostname", hostname)

	// periodically update time, steps, etc
	gui.OnRefresh(func() {
		Inject <- func() {
			// solver
			gui.SetValue("time", fmt.Sprintf("%6e", Time))
			gui.SetValue("dt", fmt.Sprintf("%4e", Solver.Dt_si))
			gui.SetValue("step", Solver.NSteps)
			gui.SetValue("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
			if pause {
				gui.SetValue("solverstatus", "paused")
			} else {
				gui.SetValue("solverstatus", "running")
			}

			// display
			// todo: use time step as cachebreaker
			cachebreaker := "?" + fmt.Sprint(time.Now().Nanosecond())
			gui.SetValue("render", "/render/"+renderQ+cachebreaker)

			// parameters
			for n, p := range params {
				v := p.GetVec()
				for comp, id := range ((*guidata)(nil)).CompBoxIds(n) {
					gui.SetValue(id, v[comp])
				}
			}

			// process
			gui.SetValue("walltime", fmt.Sprint(roundt(time.Since(StartTime))))
		}
	})

	log.Print(" =====\n open your browser and visit http://localhost", port, "\n =====\n")
	util.LogErr(http.ListenAndServe(port, nil))
	runtime.Gosched()
}

var StartTime = time.Now()

// round duration to 1s accuracy
func roundt(t time.Duration) time.Duration {
	return t - t%1e9
}

func inj(f func()) func() {
	return func() { Inject <- f }
}
