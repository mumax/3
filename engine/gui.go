package engine

import (
	"fmt"
	"github.com/barnex/cuda5/cu"
	"github.com/barnex/gui"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"log"
	"math/rand"
	"net/http"
	"path"
	"reflect"
	"sync"
	"time"
)

// global GUI state stores what is currently shown in the web page.
var GUI = guistate{Quants: make(map[string]Slicer), Params: make(map[string]Param)}

type guistate struct {
	*gui.Page
	Quants             map[string]Slicer
	Params             map[string]Param
	mutex              sync.Mutex
	_eventCacheBreaker int // changed on any event to make sure display is updated
	busy               bool
}

// displayable quantity in GUI Parameters section
type Param interface {
	NComp() int
	Name() string
	Unit() string
	getRegion(int) []float64
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
	g.Page = gui.NewPage(templText, g)
	g.OnAnyEvent(func() {
		g.incCacheBreaker()
	})

	http.Handle("/", g)
	http.Handle("/render/", &renderer)
	//http.HandleFunc("/plot/", servePlot)

	g.prepareConsole()
	g.prepareMesh()
	g.prepareGeom()
	g.prepareM()
	g.prepareSolver()
	g.prepareDisplay()
	g.prepareParam()
	g.prepareOnUpdate()
}

func (g *guistate) prepareConsole() {
	g.OnEvent("cli", func() {
		cmd := g.StringValue("cli")
		Inject <- func() { Eval(cmd) }
		g.Set("cli", "")
	})
}

func (g *guistate) prepareMesh() {
	g.Disable("setmesh", true) // button only enabled if pressing makes sense
	const MESHWARN = "&#x26a0; Click to update mesh (may take some time)"
	meshboxes := []string{"nx", "ny", "nz", "cx", "cy", "cz", "px", "py", "pz"}
	warnmesh := func() {
		g.Disable("setmesh", false)
		g.Set("setmeshwarn", MESHWARN)
	}
	for _, e := range meshboxes {
		g.OnEvent(e, warnmesh)
	}

	g.OnEvent("setmesh", func() {
		g.Disable("setmesh", true)
		Inject <- (func() {
			Eval(fmt.Sprintf("SetMesh(%v, %v, %v, %v, %v, %v, %v, %v, %v)",
				g.Value("nx"),
				g.Value("ny"),
				g.Value("nz"),
				g.Value("cx"),
				g.Value("cy"),
				g.Value("cz"),
				g.Value("px"),
				g.Value("py"),
				g.Value("pz")))
		})
		g.Set("setmeshwarn", "mesh up to date")
	})
}

func (g *guistate) prepareGeom() {
	g.OnEvent("geomselect", func() {
		ident := g.StringValue("geomselect")
		t := World.Resolve(ident).Type()
		// set sensible args: world size
		args := "("
		for i := 0; i < t.NumIn(); i++ {
			val := 0.0
			if i < 3 {
				val = Mesh().WorldSize()[i]
			}
			if i > 0 {
				args += ", "
			}
			args += fmt.Sprint(val)
		}
		args += ")"
		// overwrite args for special cases
		switch {
		case ident == "cell":
			args = "(0, 0, 0)"
		case ident == "xrange" || ident == "yrange" || ident == "zrange":
			args = "(0, inf)"
		case ident == "layers":
			args = "(0, 1)"
		}
		g.Set("geomargs", args)
		g.Set("geomdoc", g.Doc(ident))
	})
	g.OnEvent("setgeom", func() {
		Inject <- (func() {
			Eval(fmt.Sprint("SetGeom(", g.StringValue("geomselect"), g.StringValue("geomargs"), ")"))
		})
	})
}

func (g *guistate) prepareM() {
	g.OnEvent("mselect", func() {
		ident := g.StringValue("mselect")
		t := World.Resolve(ident).Type()
		args := "("
		for i := 0; i < t.NumIn(); i++ {
			val := "0"
			if i%3 == 0 {
				val = "1"
			}
			if i > 0 {
				args += ", "
			}
			args += val
		}
		args += ")"
		// overwrite args for special cases
		switch ident {
		case "vortex":
			args = "(1, 1)"
		case "vortexwall":
			args = "(1, -1, 1, 1)"
		}
		g.Set("margs", args)
		g.Set("mdoc", g.Doc(ident))
	})
	g.OnEvent("setm", func() {
		Inject <- (func() {
			Eval(fmt.Sprint("m = ", g.StringValue("mselect"), g.StringValue("margs")))
		})
	})
}

var (
	solvertypes = map[string]string{"euler": "1", "heun": "2"}
	solvernames = map[int]string{1: "euler", 2: "heun"}
)

func (g *guistate) prepareSolver() {
	g.OnEvent("run", g.cmd("Run", "runtime"))
	g.OnEvent("steps", g.cmd("Steps", "runsteps"))
	g.OnEvent("break", func() { Inject <- func() { pause = true } })
	g.OnEvent("mindt", func() { Inject <- func() { Eval("MinDt=" + g.StringValue("mindt")) } })
	g.OnEvent("maxdt", func() { Inject <- func() { Eval("MaxDt=" + g.StringValue("maxdt")) } })
	g.OnEvent("fixdt", func() { Inject <- func() { Eval("FixDt=" + g.StringValue("fixdt")) } })
	g.OnEvent("maxerr", func() { Inject <- func() { Eval("MaxErr=" + g.StringValue("maxerr")) } })
	g.OnEvent("solvertype", func() {
		Inject <- func() {
			typ := solvertypes[g.StringValue("solvertype")]
			Eval("SetSolver(" + typ + ")")
			if Solver.FixDt == 0 { // euler must have fixed time step
				Solver.FixDt = 1e-15
			}
		}
	})
}

func (g *guistate) prepareParam() {
	for _, p := range g.Params {
		p := p
		n := p.Name()
		g.OnEvent(n, func() {
			cmd := p.Name()
			r := g.Value("region")
			if r == -1 {
				cmd += " = "
			} else {
				cmd += fmt.Sprint(".SetRegion(", r, ", ")
			}
			if p.NComp() == 3 {
				cmd += "vector " // space needed
			}
			cmd += g.StringValue(p.Name())
			if r != -1 {
				cmd += ")"
			}
			Inject <- func() {
				Eval(cmd)
			}
		})
	}
	g.OnEvent("Temp", func() {
		Inject <- func() {
			if solvertype != 1 {
				Eval("SetSolver(1)")
			}
			if Solver.FixDt == 0 {
				Eval("FixDt = 1e-15")
			}
			Eval("Temp = " + g.StringValue("Temp"))
		}
	})
}

func (g *guistate) prepareDisplay() {
	g.OnEvent("renderQuant", func() {
		g.Set("renderDoc", g.Doc(g.StringValue("renderQuant")))
	})
}

func (g *guistate) prepareOnUpdate() {
	g.OnUpdate(func() {
		Req(1)
		defer Req(-1)
		updateKeepAlive() // keep track of when browser was last seen alive

		if g.Busy() {
			g.disableControls(true)
			log.Print(".")
			return
		} else {
			g.disableControls(false) // make sure everything is enabled
		}

		Inject <- (func() {
			// solver
			g.Set("nsteps", Solver.NSteps)
			g.Set("time", fmt.Sprintf("%6e", Time))
			g.Set("dt", fmt.Sprintf("%4e", Solver.Dt_si))
			g.Set("lasterr", fmt.Sprintf("%3e", Solver.LastErr))
			g.Set("maxerr", Solver.MaxErr)
			g.Set("mindt", Solver.MinDt)
			g.Set("maxdt", Solver.MaxDt)
			g.Set("fixdt", Solver.FixDt)
			g.Set("solvertype", solvernames[solvertype])
			if pause {
				g.Set("busy", "Paused")
			} else {
				g.Set("busy", "Running")
			}

			// display
			quant := g.StringValue("renderQuant")
			comp := g.StringValue("renderComp")
			cachebreaker := "?" + g.StringValue("nsteps") + "_" + fmt.Sprint(g.cacheBreaker())
			g.Set("display", "/render/"+quant+"/"+comp+cachebreaker)

			// parameters
			for _, p := range g.Params {
				n := p.Name()
				r := g.IntValue("region")
				if r == -1 && !p.IsUniform() {
					g.Set(n, "")
				} else {
					if r == -1 {
						r = 0 // uniform, so pick one
					}
					v := p.getRegion(r)
					if p.NComp() == 1 {
						g.Set(n, float32(v[0]))
					} else {
						g.Set(n, fmt.Sprintf("(%v, %v, %v)", float32(v[X]), float32(v[Y]), float32(v[Z])))
					}
				}
			}

			// gpu
			memfree, _ := cu.MemGetInfo()
			memfree /= (1024 * 1024)
			g.Set("memfree", memfree)
		})
	})
}

func (g *guistate) Doc(quant string) string {
	doc, ok := World.Doc[quant]
	if !ok {
		log.Println("no doc for", quant)
	}
	return doc
}

// returns func that injects func that executes cmd(args),
// with args ids for GUI element values.
// TODO: rm
func (g *guistate) cmd(cmd string, args ...string) func() {
	return func() {
		Inject <- func() {
			code := cmd + "("
			if len(args) > 0 {
				code += g.StringValue(args[0])
			}
			for i := 1; i < len(args); i++ {
				code += ", " + g.StringValue(args[i])
			}
			code += ")"
			Eval(code)
		}
	}
}

// todo: rm?
//func (g *guistate) floatValues(id ...string) []float64 {
//	v := make([]float64, len(id))
//	for i := range id {
//		v[i] = g.FloatValue(id[i])
//	}
//	return v
//}
//
//// todo: rm?
//func (g *guistate) intValues(id ...string) []int {
//	v := make([]int, len(id))
//	for i := range id {
//		v[i] = g.IntValue(id[i])
//	}
//	return v
//}

// renders page title for PrepareServer
func (g *guistate) Title() string   { return util.NoExt(path.Base(OD)) }
func (g *guistate) Version() string { return UNAME }
func (g *guistate) GPUInfo() string { return cuda.GPUInfo }

func (g *guistate) incCacheBreaker() {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	g._eventCacheBreaker++
}

func (g *guistate) cacheBreaker() int {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g._eventCacheBreaker
}

func (g *guistate) QuantNames() []string {
	names := make([]string, 0, len(g.Quants))
	for k, _ := range g.Quants {
		names = append(names, k)
	}
	sortNoCase(names)
	return names
}

// List all available shapes
func (g *guistate) Shapes() []string  { return g.apifilter("Shape") }
func (g *guistate) Configs() []string { return g.apifilter("Config") }

// List all api functions that return outputtype (Shape, Config, ...)
func (g *guistate) apifilter(outputtype string) []string {
	var match []string
	for k, v := range World.Identifiers {
		t := v.Type()
		if t.Kind() == reflect.Func && t.NumOut() == 1 && t.Out(0).Name() == outputtype {
			match = append(match, k)
		}
	}
	sortNoCase(match)
	return match
}

func (g *guistate) Parameters() []string {
	var params []string
	for _, v := range g.Params {
		params = append(params, v.Name())
	}
	sortNoCase(params)
	return params
}

func (g *guistate) UnitOf(quant string) string {
	p := g.Params[quant]
	if p != nil {
		return p.Unit()
	} else {
		return ""
	}
}

// renders a <div> that toggles visibility on click for PrepareServer
func (g *guistate) Div(heading string) string {
	id := fmt.Sprint("div_", rand.Int())
	return fmt.Sprintf(`<span style="cursor:pointer; font-size:1.2em; font-weight:bold; color:gray" onclick="toggle('%v')">&dtrif; %v</span> <br/> <div id="%v">`, id, heading, id)
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
	g.disableControls(busy)
	updateKeepAlive() // needed after long busy period to avoid browser considered disconnected
	if busy {
		GUI.Set("busy", "Initializing")
	} else {
		GUI.Set("busy", "")
		GUI.Set("progress", 0)
	}
}

func init() {
	util.Progress_ = GUI.Prog
}

func (g *guistate) Prog(a, total int, msg string) {
	g.Set("progress", (a*100)/total)
	g.Set("busy", msg)
	//visible := (a!=total)
	//g.Display("progress", visible)
}

func (g *guistate) disableControls(busy bool) {
	g.Disable("cli", busy)
	g.Disable("run", busy)
	g.Disable("steps", busy)
	g.Disable("break", busy)
}

func (g *guistate) Busy() bool {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.busy
}

func Eval(code string) {
	tree, err := World.Compile(code)
	if err == nil {
		LogInput(rmln(tree.Format()))
		tree.Eval()
	} else {
		LogOutput(code + "\n" + err.Error())
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
	keepalive = time.Now()
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
