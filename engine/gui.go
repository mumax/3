package engine

import (
	"fmt"
	"github.com/barnex/gui"
	"github.com/mumax/3/util"
	"math/rand"
	"net/http"
	"path"
	"sync"
	"time"
)

// global GUI state stores what is currently shown in the web page.
var GUI = guistate{Quants: make(map[string]Slicer), Params: make(map[string]Param)}

type guistate struct {
	*gui.Page
	Quants map[string]Slicer
	Params map[string]Param
	mutex  sync.Mutex
	busy   bool
}

// displayable quantity in GUI Parameters section
type Param interface {
	NComp() int
	Unit() string
	getRegion(int) []float64
	setRegion(int, []float64)
	IsUniform() bool
}

// Internal:add a quantity to the GUI, will be visible in web interface.
// Automatically called by Decl*(), still before PrepareServer()
func (g *guistate) Add(name string, value interface{}) {
	if v, ok := value.(Param); ok {
		g.Params[name] = v
	}
	if v, ok := value.(Slicer); ok {
		g.Quants[name] = v
	}
}

// Once Params/Quants have been declared and added,
// initialize the GUI Page (pre-renders template) and register http handlers
func (g *guistate) PrepareServer() {
	GUI.Page = gui.NewPage(templText, &GUI)

	http.Handle("/", GUI)
	//http.HandleFunc("/render/", serveRender)
	//http.HandleFunc("/plot/", servePlot)

	// console
	GUI.OnEvent("cli", func() {
		cmd := GUI.StringValue("cli")
		Inject <- func() { Eval(cmd) }
		GUI.Set("cli", "")
	})

	// geometry
	GUI.OnEvent("setmesh", func() {

	})

	GUI.OnUpdate(func() {
		updateKeepAlive() // keep track of when browser was last seen alive
	})
}

// renders page title for PrepareServer
func (g *guistate) Title() string {
	return util.NoExt(path.Base(OD))
}

// renders a <div> that toggles visibility on click for PrepareServer
func (g *guistate) Div(heading string) string {
	id := fmt.Sprint("div_", rand.Int())
	return fmt.Sprintf(`<h2 style="cursor:pointer" onclick="toggle('%v')">&dtrif; %v</h2> <div id="%v">`, id, heading, id)
}

// Start web gui on given port, blocks.
func Serve(port string) {
	util.LogErr(http.ListenAndServe(port, nil))
}

// When gui is busy it can only accept read-only
// commands, not change any state. E.g. during kernel init.
func (g *guistate) SetBusy(busy bool) {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	g.busy = busy
	GUI.Disable("cli", busy)
	GUI.Disable("setmesh", busy)
}

func Eval(code string) {
	tree, err := World.Compile(code)
	if err == nil {
		Log(tree.Format())
		tree.Eval()
	} else {
		Log(err)
	}
}

//
//var (
//	gui_           *gui.Page
//	renderQ        = "m"                     // quantity to display
//	guiRegion      = -1                      // currently addressed region
//	usingX, usingY = 1, 2                    // columns to plot
//	busyMsg        string                    // set busy message here when doing slow initialization
//)
//
//const maxZoom = 32
//
//// data for html template
//type guidata struct {
//	Quants map[string]Slicer
//	Params map[string]Param
//}
//
//func SetBusy(msg string) {
//	//guiLock.Lock()
//	//defer guiLock.Unlock()
//	busyMsg = msg
//}
//
//func busy() string {
//	//guiLock.Lock()
//	//defer guiLock.Unlock()
//	return busyMsg
//}
//
//// list of text box id's for component text boxes.
//func (d *guidata) CompBoxIds(param string) []string {
//	var e []string
//	p := params[param]
//	for i := 0; i < p.NComp(); i++ {
//		e = append(e, fmt.Sprint("param_", param, i))
//	}
//	return e
//}
//
//// util for generating region numbers in GUI Parameters section.
//func (d *guidata) MakeRange(min, max int) []int {
//	l := make([]int, max-min)
//	for i := range l {
//		l[i] = min + i
//	}
//	return l
//}
//
//
//func InitGui() {
//	data := &guidata{Quants: quants, Params: params}
//	gui_ = gui.NewPage(templText, data)
//	gui := gui_
//
//	http.Handle("/", gui)
//	http.HandleFunc("/render/", serveRender)
//	http.HandleFunc("/plot/", servePlot)
//
//	// solver
//	gui.OnEvent("break", inj(func() { pause = true }))
//	gui.OnEvent("run", inj(func() { Run(gui.Value("runtime").(float64)) }))
//	gui.OnEvent("steps", inj(func() { Steps(gui.Value("runsteps").(int)) }))
//	gui.OnEvent("fixdt", inj(func() { Solver.FixDt = gui.Value("fixdt").(float64) }))
//	gui.OnEvent("mindt", inj(func() { Solver.MinDt = gui.Value("mindt").(float64) }))
//	gui.OnEvent("maxdt", inj(func() { Solver.MaxDt = gui.Value("maxdt").(float64) }))
//	gui.OnEvent("maxerr", inj(func() { Solver.MaxErr = gui.Value("maxerr").(float64) }))
//	gui.OnEvent("sel_render", func() { renderQ = gui.Value("sel_render").(string) })
//	gui.OnEvent("renderComp", func() { renderComp = gui.Value("renderComp").(string) })
//	gui.OnEvent("renderScale", func() { renderScale = (maxZoom + 1) - gui.Value("renderScale").(int) })
//	gui.OnEvent("renderLayer", func() { renderLayer = gui.Value("renderLayer").(int) })
//	gui.OnEvent("command", handleCommand)
//
//	// display
//	gui.Set("sel_render", renderQ)
//
//	// gnuplot
//	gui.OnEvent("usingX", func() { usingX = gui.Value("usingX").(int) })
//	gui.OnEvent("usingY", func() { usingY = gui.Value("usingY").(int) })
//
//	// setting parameters
//	gui.Set("sel_region", guiRegion)
//	gui.OnEvent("sel_region", func() { guiRegion = atoi(gui.Value("sel_region")) })
//
//	for n, p := range params {
//		n := n // closure caveats...
//		p := p
//
//		compIds := ((*guidata)(nil)).CompBoxIds(n)
//		handler := func() {
//			var cmd string
//			if guiRegion == -1 {
//				cmd = fmt.Sprintf("%v = (", n)
//			} else {
//				cmd = fmt.Sprintf("%v.setRegion(%v,", n, guiRegion)
//			}
//			if p.NComp() == 3 {
//				cmd += fmt.Sprintf("vector(%v, %v, %v)",
//					gui.Value(compIds[0]), gui.Value(compIds[1]), gui.Value(compIds[2]))
//			} else {
//				cmd += fmt.Sprint(gui.Value(compIds[0]))
//			}
//			cmd += ");"
//			Inject <- func() { Eval(cmd) } // Eval(cmd) can be very long, launch but don't wait
//		}
//		for _, id := range compIds {
//			gui.OnEvent(id, handler)
//		}
//	}
//
//	// process
//	gui.Set("gpu", fmt.Sprint(cuda.DevName, " (", (cuda.TotalMem)/(1024*1024), "MB)", ", CUDA ", cuda.Version))
//	hostname, _ := os.Hostname()
//	gui.Set("hostname", hostname)
//	var memstats runtime.MemStats
//
//	// periodically update time, steps, etc
//	onrefresh := func() {
//
//		updateKeepAlive()
//		gui.Set("hist", hist)
//
//		// geometry
//		size := globalmesh.Size()
//		gui.Set("nx", size[0])
//		gui.Set("ny", size[1])
//		gui.Set("nz", size[2])
//		cellSize := globalmesh.CellSize()
//		gui.Set("cx", float32(cellSize[0]*1e9)) // in nm
//		gui.Set("cy", float32(cellSize[1]*1e9))
//		gui.Set("cz", float32(cellSize[2]*1e9))
//		gui.Set("wx", float32(float64(size[0])*cellSize[0]*1e9))
//		gui.Set("wy", float32(float64(size[1])*cellSize[1]*1e9))
//		gui.Set("wz", float32(float64(size[2])*cellSize[2]*1e9))
//
//		// solver
//		gui.Set("time", fmt.Sprintf("%6e", Time))
//		gui.Set("dt", fmt.Sprintf("%4e", Solver.Dt_si))
//		gui.Set("step", Solver.NSteps)
//		gui.Set("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
//		gui.Set("maxerr", Solver.MaxErr)
//		gui.Set("mindt", Solver.MinDt)
//		gui.Set("maxdt", Solver.MaxDt)
//		gui.Set("fixdt", Solver.FixDt)
//		if pause {
//			gui.Set("solverstatus", "paused")
//		} else {
//			gui.Set("solverstatus", "running")
//		}
//
//		// display
//		cachebreaker := fmt.Sprint("?", Solver.NSteps, renderScale) // scale needed if we zoom while paused
//		gui.Set("render", "/render/"+renderQ+cachebreaker)
//		gui.Set("renderComp", renderComp)
//		gui.Set("renderLayer", renderLayer)
//		gui.Set("renderScale", (maxZoom+1)-renderScale)
//
//		// plot
//		cachebreaker = fmt.Sprint("?", Solver.NSteps)
//		gui.Set("plot", "/plot/"+cachebreaker)
//
//		// parameters
//		for n, p := range params {
//			if guiRegion == -1 {
//				if p.IsUniform() {
//					v := p.getRegion(0)
//					for comp, id := range ((*guidata)(nil)).CompBoxIds(n) {
//						gui.Set(id, fmt.Sprintf("%g", float32(v[comp])))
//					}
//				} else {
//					for _, id := range ((*guidata)(nil)).CompBoxIds(n) {
//						gui.Set(id, "")
//					}
//				}
//			} else {
//				v := p.getRegion(guiRegion)
//				for comp, id := range ((*guidata)(nil)).CompBoxIds(n) {
//					gui.Set(id, fmt.Sprintf("%g", float32(v[comp])))
//				}
//			}
//		}
//
//		// process
//		gui.Set("walltime", fmt.Sprint(roundt(time.Since(StartTime))))
//		runtime.ReadMemStats(&memstats)
//		gui.Set("memstats", memstats.TotalAlloc/(1024))
//	}
//
//	gui.OnUpdate(func() {
//		// do not inject into run() loop if we are very busy doing other stuff
//		busy := busy()
//		if busy != "" {
//			gui.Set("solverstatus", fmt.Sprint(busy)) // show we are busy, ignore the rest
//		} else {
//			InjectAndWait(onrefresh) // onrefresh is fast (just fetches values), so wait
//		}
//	})
//
//}
//
//
//// round duration to 1s accuracy
//func roundt(t time.Duration) time.Duration {
//	return t - t%1e9
//}
//
//// returns a function that injects f into run loop
//func inj(f func()) func() {
//	return func() { Inject <- f }
//}
//
//func Eval(code string) {
//	defer func() {
//		err := recover()
//		if err != nil {
//			gui_.Set("solverstatus", fmt.Sprint(err)) // TODO: not solverstatus
//			util.Log(err)
//		}
//	}()
//	tree, err := World.Compile(code)
//	if err == nil {
//		Log(tree.Format())
//		tree.Eval()
//	} else {
//		gui_.Set("paramErr", fmt.Sprint(err))
//		util.Log(err)
//	}
//}
//
//// TODO: unify with Eval
//func handleCommand() {
//	gui := gui_
//	command := gui.Value("command").(string)
//	Inject <- func() {
//		tree, err := World.Compile(command)
//		if err != nil {
//			gui.Set("cmderr", fmt.Sprint(err))
//			return
//		}
//		Log(tree.Format())
//		gui.Set("command", "")
//		tree.Eval()
//		gui.Set("cmderr", "")
//	}
//}
//
//func atoi(x interface{}) int {
//	i, err := strconv.Atoi(fmt.Sprint(x))
//	util.LogErr(err)
//	return i
//}

var (
	keepalive time.Time
	keepalock sync.Mutex
)

func KeepAlive() time.Time {
	keepalock.Lock()
	defer keepalock.Unlock()
	return keepalive
}

func updateKeepAlive() {
	keepalock.Lock()
	defer keepalock.Unlock()
	keepalive = time.Now()
}
