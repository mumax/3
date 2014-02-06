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
var (
	GUI           = guistate{Quants: make(map[string]Quantity), Params: make(map[string]Param)}
	keepalive     = time.Now()
	keepaliveLock sync.Mutex
)

// Returns the time when updateKeepAlive was called.
func KeepAlive() time.Time {
	keepaliveLock.Lock()
	defer keepaliveLock.Unlock()
	return keepalive
}

// Called on each http request to signal browser is still open.
func updateKeepAlive() {
	keepaliveLock.Lock()
	defer keepaliveLock.Unlock()
	keepalive = time.Now()
}

type guistate struct {
	*gui.Page                              // GUI elements (buttons...)
	Quants             map[string]Quantity // displayable quantities by name
	Params             map[string]Param    // displayable parameters by name
	mutex              sync.Mutex          // protects eventCacheBreaker and busy
	_eventCacheBreaker int                 // changed on any event to make sure display is updated
	busy               bool                // are we so busy we can't respond from run loop? (e.g. calc kernel)
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
	if v, ok := value.(Quantity); ok {
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
	http.HandleFunc("/plot/", servePlot)

	g.prepareConsole()
	g.prepareMesh()
	g.prepareGeom()
	g.prepareM()
	g.prepareSolver()
	g.prepareDisplay()
	g.prepareParam()
	g.prepareOnUpdate()
}

// see prepareServer
func (g *guistate) prepareConsole() {
	g.OnEvent("cli", func() {
		cmd := g.StringValue("cli")
		Inject <- func() { Eval(cmd) }
		g.Set("cli", "")
	})
}

// see prepareServer
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
				g.Value("nx"), g.Value("ny"), g.Value("nz"),
				g.Value("cx"), g.Value("cy"), g.Value("cz"),
				g.Value("px"), g.Value("py"), g.Value("pz")))
		})
		g.Set("setmeshwarn", "mesh up to date")
	})
}

// see prepareServer
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
		case ident == "Cell":
			args = "(0, 0, 0)"
		case ident == "XRange" || ident == "YRange" || ident == "ZRange":
			args = "(0, inf)"
		case ident == "Layers":
			args = "(0, 1)"
		case ident == "ImageShape":
			args = `("filename.png")`
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

// see prepareServer
func (g *guistate) prepareM() {
	g.OnEvent("mselect", func() {
		ident := g.StringValue("mselect")
		t := World.Resolve(ident).Type()
		args := "("
		for i := 0; i < t.NumIn(); i++ {
			if i > 0 {
				args += ", "
			}
			args += "1"
		}
		args += ")"
		// overwrite args for special cases
		switch ident {
		case "VortexWall":
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

// see prepareServer
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

// see prepareServer
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

// see prepareServer
func (g *guistate) prepareDisplay() {
	g.OnEvent("renderQuant", func() {
		g.Set("renderDoc", g.Doc(g.StringValue("renderQuant")))
	})
}

// see prepareServer
func (g *guistate) prepareOnUpdate() {
	g.OnUpdate(func() {
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
			if !pause { // Don't go updating stuff while paused
				g.Set("maxtorque", fmt.Sprintf("%6e T", MaxTorque.Get()))
			}
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

			// plot
			GUI.Set("plot", "/plot/?"+cachebreaker)

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

// Returns documentation string for quantity name. E.g.:
// 	"m" -> "Reduced magnetization"
func (g *guistate) Doc(quant string) string {
	doc, ok := World.Doc[quant]
	if !ok {
		log.Println("no doc for", quant)
	}
	return doc
}

// Returns unit for quantity name. E.g.:
// 	"Msat" -> "A/m"
func (g *guistate) UnitOf(quant string) string {
	p := g.Params[quant]
	if p != nil {
		return p.Unit()
	} else {
		return ""
	}
}

// returns func that injects func that executes cmd(args),
// with args ids for GUI element values.
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
	for k, _ := range World.Doc {
		v := World.Resolve(k)
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

// renders a <div> that toggles visibility on click for PrepareServer
func (g *guistate) Div(heading string) string {
	id := fmt.Sprint("div_", rand.Int())
	return fmt.Sprintf(`<span title="Click to show/hide" style="cursor:pointer; font-size:1.2em; font-weight:bold; color:gray" onclick="toggle('%v')">&dtrif; %v</span> <br/> <div id="%v">`, id, heading, id)
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

// Prog advances the GUI progress bar to fraction a/total and displays message.
func (g *guistate) Prog(a, total int, msg string) {
	g.Set("progress", (a*100)/total)
	g.Set("busy", msg)
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
//// round duration to 1s accuracy
//func roundt(t time.Duration) time.Duration {
//	return t - t%1e9
//}
//
