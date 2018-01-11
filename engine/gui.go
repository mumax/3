package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/gui"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"math/rand"
	"net"
	"net/http"
	"path"
	"reflect"
	"strconv"
	"sync"
	"time"
)

// global GUI state stores what is currently shown in the web page.
var (
	gui_    = guistate{Quants: make(map[string]Quantity), Params: make(map[string]Param)}
	Timeout = 3 * time.Second // exit finished simulation this long after browser was closed
)

type guistate struct {
	*gui.Page                              // GUI elements (buttons...)
	Quants             map[string]Quantity // displayable quantities by name
	Params             map[string]Param    // displayable parameters by name
	render                                 // renders displayed quantity
	mutex              sync.Mutex          // protects eventCacheBreaker and keepalive
	_eventCacheBreaker int                 // changed on any event to make sure display is updated
	keepalive          time.Time
}

// Returns the time when updateKeepAlive was called.
func (g *guistate) KeepAlive() time.Time {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	return g.keepalive
}

// Called on each http request to signal browser is still open.
func (g *guistate) UpdateKeepAlive() {
	g.mutex.Lock()
	defer g.mutex.Unlock()
	g.keepalive = time.Now()
}

func nop() {}

// Enter interactive mode. Simulation is now exclusively controlled by web GUI
func (g *guistate) RunInteractive() {

	// periodically wake up Run so it may exit on timeout
	go func() {
		for {
			Inject <- nop
			time.Sleep(1 * time.Second)
		}
	}()

	fmt.Println("//entering interactive mode")
	g.UpdateKeepAlive()
	for time.Since(g.KeepAlive()) < Timeout {
		f := <-Inject
		f()
	}
	fmt.Println("//browser disconnected, exiting")
}

// displayable quantity in GUI Parameters section
type Param interface {
	NComp() int
	Name() string
	Unit() string
	getRegion(int) []float64
	IsUniform() bool
}

func GUIAdd(name string, value interface{}) {
	gui_.Add(name, value)
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
	util.SetProgress(gui_.Prog)
	g.OnAnyEvent(func() {
		g.incCacheBreaker()
	})

	http.Handle("/", g)
	http.HandleFunc("/render/", g.ServeRender)
	http.HandleFunc("/plot/", g.servePlot)

	g.Set("title", util.NoExt(OD()[:len(OD())-1]))
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
		Inject <- func() { g.EvalGUI(cmd) }
		g.Set("cli", "")
	})
}

// see prepareServer
func (g *guistate) prepareMesh() {
	//g.Disable("setmesh", true) // button only enabled if pressing makes sense
	const MESHWARN = "&#x26a0; Click to update mesh (may take some time)"

	warnmesh := func() {
		//g.Disable("setmesh", false)
		g.Set("setmeshwarn", MESHWARN)
	}

	g.OnEvent("nx", func() { Inject <- func() { lazy_gridsize[X] = g.IntValue("nx"); warnmesh() } })
	g.OnEvent("ny", func() { Inject <- func() { lazy_gridsize[Y] = g.IntValue("ny"); warnmesh() } })
	g.OnEvent("nz", func() { Inject <- func() { lazy_gridsize[Z] = g.IntValue("nz"); warnmesh() } })
	g.OnEvent("cx", func() { Inject <- func() { lazy_cellsize[X] = g.FloatValue("cx"); warnmesh() } })
	g.OnEvent("cy", func() { Inject <- func() { lazy_cellsize[Y] = g.FloatValue("cy"); warnmesh() } })
	g.OnEvent("cz", func() { Inject <- func() { lazy_cellsize[Z] = g.FloatValue("cz"); warnmesh() } })
	g.OnEvent("px", func() { Inject <- func() { lazy_pbc[X] = g.IntValue("px"); warnmesh() } })
	g.OnEvent("py", func() { Inject <- func() { lazy_pbc[Y] = g.IntValue("py"); warnmesh() } })
	g.OnEvent("pz", func() { Inject <- func() { lazy_pbc[Z] = g.IntValue("pz"); warnmesh() } })

	g.OnEvent("setmesh", func() {
		//g.Disable("setmesh", true)
		Inject <- (func() {
			g.EvalGUI(fmt.Sprintf("SetMesh(%v, %v, %v, %v, %v, %v, %v, %v, %v)",
				g.Value("nx"), g.Value("ny"), g.Value("nz"),
				g.Value("cx"), g.Value("cy"), g.Value("cz"),
				g.Value("px"), g.Value("py"), g.Value("pz")))
			// update lazy_* sizes to be up-to date with proper mesh
			n := Mesh().Size()
			c := Mesh().CellSize()
			p := Mesh().PBC()
			lazy_gridsize = []int{n[X], n[Y], n[Z]}
			lazy_cellsize = []float64{c[X], c[Y], c[Z]}
			lazy_pbc = []int{p[X], p[Y], p[Z]}

		})
		g.Set("setmeshwarn", "mesh up to date")
	})
}

func (g *guistate) IntValue(id string) int {
	s := g.StringValue(id)
	r := fmt.Sprint(Eval1Line(s))
	i, _ := strconv.Atoi(r)
	return i
}

func (g *guistate) FloatValue(id string) float64 {
	s := g.StringValue(id)
	r := fmt.Sprint(Eval1Line(s))
	f, _ := strconv.ParseFloat(r, 64)
	return f
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
			g.EvalGUI(fmt.Sprint("SetGeom(", g.StringValue("geomselect"), g.StringValue("geomargs"), ")"))
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
			g.EvalGUI(fmt.Sprint("m = ", g.StringValue("mselect"), g.StringValue("margs")))
		})
	})
}

var (
	solvertypes = map[string]int{"bw_euler": -1, "euler": 1, "heun": 2, "rk23": 3, "rk4": 4, "rk45": 5, "rkf56": 6}
	solvernames = map[int]string{-1: "bw_euler", 1: "euler", 2: "heun", 3: "rk23", 4: "rk4", 5: "rk45", 6: "rkf56"}
)

func Break() {
	Inject <- func() { pause = true }
}

// see prepareServer
func (g *guistate) prepareSolver() {
	g.OnEvent("run", func() { Break(); Inject <- func() { g.EvalGUI(sprint("Run(", g.StringValue("runtime"), ")")) } })
	g.OnEvent("steps", func() { Break(); Inject <- func() { g.EvalGUI(sprint("Steps(", g.StringValue("runsteps"), ")")) } })
	g.OnEvent("break", Break)
	g.OnEvent("relax", func() { Break(); Inject <- func() { g.EvalGUI("relax()") } })
	g.OnEvent("mindt", func() { Inject <- func() { g.EvalGUI("MinDt=" + g.StringValue("mindt")) } })
	g.OnEvent("maxdt", func() { Inject <- func() { g.EvalGUI("MaxDt=" + g.StringValue("maxdt")) } })
	g.OnEvent("fixdt", func() { Inject <- func() { g.EvalGUI("FixDt=" + g.StringValue("fixdt")) } })
	g.OnEvent("maxerr", func() { Inject <- func() { g.EvalGUI("MaxErr=" + g.StringValue("maxerr")) } })
	g.OnEvent("solvertype", func() {
		Inject <- func() {
			typ := solvertypes[g.StringValue("solvertype")]

			// euler must have fixed time step
			if typ == EULER && FixDt == 0 {
				g.EvalGUI("FixDt = 1e-15")
			}
			if typ == BACKWARD_EULER && FixDt == 0 {
				g.EvalGUI("FixDt = 1e-13")
			}

			g.EvalGUI(fmt.Sprint("SetSolver(", typ, ")"))
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
				g.EvalGUI(cmd)
			}
		})
	}
	// overwrite handler for temperature
	// do not crash when we enter bogus values (see temperature.go)
	g.OnEvent("Temp", func() {
		Inject <- func() {
			if FixDt == 0 {
				g.EvalGUI("FixDt = 10e-14") // finite temperature requires fixed time step
			}
			g.EvalGUI("Temp = " + g.StringValue("Temp"))
		}
	})
}

// see prepareServer
func (g *guistate) prepareDisplay() {
	// plot
	g.OnEvent("tableAutoSave", func() {
		Inject <- func() {
			g.EvalGUI("TableAutosave(" + g.StringValue("tableAutoSave") + ")")
		}
	})

	// render
	g.OnEvent("renderQuant", func() {
		g.render.mutex.Lock()
		defer g.render.mutex.Unlock()
		name := g.StringValue("renderQuant")
		q := g.Quants[name]
		if q == nil {
			LogErr("display: unknown quantity:", name)
			return
		}
		g.render.quant = q
		g.Set("renderDoc", g.Doc(g.StringValue("renderQuant")))
	})
	g.OnEvent("renderComp", func() {
		g.render.mutex.Lock()
		defer g.render.mutex.Unlock()
		g.render.comp = g.StringValue("renderComp")
		// TODO: set to "" if q.Ncomp < 3
	})
	g.OnEvent("renderLayer", func() {
		g.render.mutex.Lock()
		defer g.render.mutex.Unlock()
		g.render.layer = g.IntValue("renderLayer")
		g.Set("renderLayerLabel", fmt.Sprint(g.render.layer, "/", Mesh().Size()[Z]))
	})
	g.OnEvent("renderScale", func() {
		g.render.mutex.Lock()
		defer g.render.mutex.Unlock()
		g.render.scale = maxScale - g.IntValue("renderScale")
		g.Set("renderScaleLabel", fmt.Sprint("1/", g.render.scale))
	})
}

// see prepareServer
func (g *guistate) prepareOnUpdate() {
	g.OnUpdate(func() {
		g.UpdateKeepAlive() // keep track of when browser was last seen alive

		if GetBusy() { // busy, e.g., calculating kernel, run loop will not accept commands.
			return
		}

		Inject <- (func() { // sends to run loop to be executed in between time steps
			g.Set("console", hist)

			// mesh
			g.Set("nx", lazy_gridsize[X])
			g.Set("ny", lazy_gridsize[Y])
			g.Set("nz", lazy_gridsize[Z])
			g.Set("cx", lazy_cellsize[X])
			g.Set("cy", lazy_cellsize[Y])
			g.Set("cz", lazy_cellsize[Z])
			g.Set("px", lazy_pbc[X])
			g.Set("py", lazy_pbc[Y])
			g.Set("pz", lazy_pbc[Z])
			g.Set("wx", printf(lazy_cellsize[X]*float64(lazy_gridsize[X])*1e9))
			g.Set("wy", printf(lazy_cellsize[Y]*float64(lazy_gridsize[Y])*1e9))
			g.Set("wz", printf(lazy_cellsize[Z]*float64(lazy_gridsize[Z])*1e9))

			// solver
			g.Set("nsteps", NSteps)
			g.Set("time", fmt.Sprintf("%1.5e", Time))
			g.Set("dt", fmt.Sprintf("%1.3e", Dt_si))
			g.Set("lasterr", fmt.Sprintf("%1.3e", LastErr))
			g.Set("maxerr", MaxErr)
			g.Set("mindt", MinDt)
			g.Set("maxdt", MaxDt)
			g.Set("fixdt", FixDt)
			g.Set("solvertype", fmt.Sprint(solvernames[solvertype]))
			if pause {
				g.Set("busy", "Paused")
			} else {
				g.Set("busy", "Running")
				// Don't re-evaluate all the time if not running
				g.Set("maxtorque", fmt.Sprintf("%1.3e T", LastTorque))
			}

			// display
			g.Set("tableAutoSave", Table.autosave.period)
			quant := g.StringValue("renderQuant")
			comp := g.StringValue("renderComp")
			cachebreaker := "?" + g.StringValue("nsteps") + "_" + fmt.Sprint(g.cacheBreaker())
			g.Attr("renderLayer", "max", Mesh().Size()[Z]-1)
			g.Set("display", "/render/"+quant+"/"+comp+cachebreaker)

			// plot
			gui_.Set("plot", "/plot/"+cachebreaker)

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
		LogErr("no doc for", quant)
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

// renders page title for PrepareServer
func (g *guistate) Title() string   { return util.NoExt(path.Base(OD())) }
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

func GoServe(addr string) string {
	gui_.PrepareServer()

	// find a free port starting from the usual number
	l, err := net.Listen("tcp", addr)
	for err != nil {
		h, p, _ := net.SplitHostPort(addr)
		addr = fmt.Sprint(h, ":", atoi(p)+1)
		l, err = net.Listen("tcp", addr)
	}
	go func() { LogErr(http.Serve(l, nil)) }()
	httpfs.Put(OD()+"gui", []byte(l.Addr().String()))
	return addr
}

func atoi(a string) int {
	i, err := strconv.Atoi(a)
	util.PanicErr(err)
	return i
}

// Prog advances the GUI progress bar to fraction a/total and displays message.
func (g *guistate) Prog(a, total int, msg string) {
	g.Set("progress", (a*100)/total)
	g.Set("busy", msg)
	util.PrintProgress(a, total, msg)
}

// Eval code + update keepalive in case the code runs long
func (g *guistate) EvalGUI(code string) {
	defer func() {
		if err := recover(); err != nil {
			if userErr, ok := err.(UserErr); ok {
				LogErr(userErr)
			} else {
				panic(err)
			}
		}
	}()
	Eval(code)
	g.UpdateKeepAlive()
}

//
//// round duration to 1s accuracy
//func roundt(t time.Duration) time.Duration {
//	return t - t%1e9
//}
//
