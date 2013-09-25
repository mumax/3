package main

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
	_ "github.com/mumax/3/ext"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strings"
	"text/template"
	"unicode"

//	"regexp"
)

var (
	api_entries entries
	api_ident   = make(map[string]entry)
)

type entry struct {
	name    string
	Type    reflect.Type
	Doc     string
	touched bool
}

func buildAPI() {
	cuda.Init(0, "yield") // gpu 0
	cuda.LockThread()

	ident := engine.World.Identifiers
	doc := engine.World.Doc
	e := make(entries, 0, len(ident))
	for K, v := range doc {
		k := strings.ToLower(K)
		t := ident[k].Type()
		entr := entry{K, t, v, false}
		e = append(e, &entr)
		api_ident[k] = entr
	}
	sort.Sort(&e)
	api_entries = e
}

func (e *entry) Name() string {
	return e.name
}

func (e *entry) Ins() string {
	t := e.Type.String()
	if strings.HasPrefix(t, "func(") {
		return cleanType(t[len("func"):])
	} else {
		return ""
	}
}

func cleanType(typ string) string {
	return strings.Replace(typ, "engine.", "", -1)
}

func (e *entry) Methods() []string {
	t := e.Type
	// if it's a function, we list the methods on the output type
	if t.Kind() == reflect.Func && t.NumOut() == 1 {
		t = t.Out(0)
	}
	nm := t.NumMethod()
	m := make([]string, 0, nm)
	for i := 0; i < nm; i++ {
		n := t.Method(i).Name
		if unicode.IsUpper(rune(n[0])) && !hidden(n) {
			m = append(m, n)
		}
	}
	return m
}

func (e *entry) Ret() string {
	t := e.Type
	if t.Kind() == reflect.Func && t.NumOut() == 1 {
		return cleanType(t.Out(0).String())
	} else {
		return ""
	}
}

func hidden(name string) bool {
	switch name {
	default:
		return false
	case "Eval", "InputType", "Type", "Slice":
		return true
	}
}

func (e *entry) Examples() []int {
	return api_examples[strings.ToLower(e.name)]
}

type api struct {
	Entries entries
}

func (e *api) Include(fname string) string {
	b, err := ioutil.ReadFile(fname)
	check(err)
	return string(b)
}

func (a *api) FilterType(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if e.touched {
			continue
		}
		for _, t := range typ {
			if match(t, e.Type.String()) {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

func (a *api) FilterReturn(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if e.touched {
			continue
		}
		for _, t := range typ {
			if match(t, e.Ret()) {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

func (a *api) FilterName(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if e.touched {
			continue
		}
		for _, t := range typ {
			if match(t, e.name) {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

func (a *api) FilterLeftovers() []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if e.touched {
			continue
		}
		E = append(E, e)
	}
	return E
}

func match(a, b string) bool {
	//match, err := regexp.MatchString(a, b)
	//check(err)
	a = strings.ToLower(a)
	b = strings.ToLower(b)
	match := a == b
	println("match", a, "-", b, match)
	return match
}

func renderAPI() {
	e := api_entries
	t := template.Must(template.New("api").Parse(templ))
	f, err2 := os.OpenFile("api.html", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)
	check(t.Execute(f, &api{e}))
}

const templ = `
{{define "entry"}}
	<p><span style="color:#000088; font-size:1.3em"> <b>{{.Name}}</b>{{.Ins}} </span>

	{{with .Doc}} <p> {{.}} </p> {{end}}

	{{with .Examples}} <p> <b>examples:</b> 
		{{range .}} 
			<a href="examples.html#example{{.}}">[{{.}}]</a> 
		{{end}} 
		</p> 
	{{end}}

	{{with .Methods}} 
		<p> <span style="color:grey"> <b>methods:</b> 
		{{range .}} {{.}} {{end}} 
		</span> </p> 
	{{end}}

	</p><hr/>
{{end}}


{{.Include "head.html"}}

<h1> mumax<sup>3</sup> API </h1>

<h1> Setting  the mesh size</h1>
The simulation mesh defines the maximum size of the magnet. It should be set at the beginning of the script. For the number of cells, powers of two or numbers with small prime factors are advisable. E.g.:
<pre><code>Nx := 128
Ny := 64
Nz := 2
sizeX := 500e-9
sizeY := 250e-9
sizeZ := 10e-9
SetGridSize(Nx, Ny, Nz)
SetCellSize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)
</code></pre>

{{range .FilterName "setgridsize" "setcellsize"}} {{template "entry" .}} {{end}}


<h1> Setting a geometry </h1>

Once the gridsize has been set, optionally a magnet Shape can be specified. The geometry is always constrained to the simulation box. Primitive shapes are constructed at the origin (box center) by default, and can be rotated and translated if needed. E.g.:
<pre><code> SetGeom(cylinder(400e-9, 20e-9).RotX().Transl(1e-6,0,0))
</code></pre>

{{range .FilterName "setgeom"}} {{template "entry" .}} {{end}}
Shape constructors:
{{range .FilterReturn "Shape"}} {{template "entry" .}} {{end}}


<h1> Misc </h1>
{{range .FilterLeftovers}} {{template "entry" .}} {{end}}

</body>
`

type entries []*entry

func (e *entries) Len() int {
	return len(*e)
}

func (e *entries) Less(i, j int) bool {
	return strings.ToLower((*e)[i].name) < strings.ToLower((*e)[j].name)
}

func (e *entries) Swap(i, j int) {
	(*e)[i], (*e)[j] = (*e)[j], (*e)[i]
}
