// Automatic generation of api.html based on template.
package main

import (
	"os"
	"os/exec"
	"path"
	"reflect"
	"sort"
	"strings"
	"text/template"
	"unicode"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/engine"
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
	cuda.Init(0) // gpu 0

	ident := engine.World.Identifiers
	doc := engine.World.Doc
	e := make(entries, 0, len(ident))
	for K, v := range doc {
		if v == "" { // check if we a docstring in the documantation of the Math package
			v = getGoDocString("math", K)
		}
		k := strings.ToLower(K)
		t := ident[k].Type()
		entr := entry{K, t, v, false}
		e = append(e, &entr)
		api_ident[k] = entr
	}
	sort.Sort(&e)
	api_entries = e
}

func getGoDocString(packageName, identifier string) string {
	docString := ""
	cmd := exec.Command("go", "doc", packageName, identifier)
	stdout, err := cmd.Output()
	outputLines := strings.Split(string(stdout), "\n")
	if err == nil && outputLines[2][:4] == "func" { // we only look for doc strings of functions
		// the doc string of a function is on the fourth line
		// (and possible continued on the fifth line, if not, then the fifth line is empty)
		docString = strings.Join(outputLines[3:5], " ")
	}
	return docString
}

func (e *entry) Name() string {
	return e.name
}

// input parameters
func (e *entry) Ins() string {
	t := e.Type.String()
	if strings.HasPrefix(t, "func(") {
		return cleanType(t[len("func"):])
	} else {
		return ""
	}
}

// dumbed-down type
func cleanType(typ string) string {
	typ = strings.Replace(typ, "engine.", "", -1)
	typ = strings.Replace(typ, "*data.", "", -1)
	typ = strings.Replace(typ, "script.", "", -1)
	return typ
}

func (e *entry) Methods() []string {
	t := e.Type
	// if it's a function, we list the methods on the output type
	if t.Kind() == reflect.Func && t.NumOut() == 1 {
		t = t.Out(0)
	}
	nm := t.NumMethod()
	M := make([]string, 0, nm)
	for i := 0; i < nm; i++ {
		m := t.Method(i)
		n := m.Name
		if unicode.IsUpper(rune(n[0])) && !hidden(n) {
			var args string
			for i := 1; i < m.Type.NumIn(); i++ {
				args += cleanType(m.Type.In(i).String()) + " "
			}
			M = append(M, n+"( "+args+")")
		}
	}
	return M
}

// return value
func (e *entry) Ret() string {
	t := e.Type
	if t.Kind() == reflect.Func && t.NumOut() == 1 {
		return cleanType(t.Out(0).String())
	} else {
		return ""
	}
}

// hidden methods
func hidden(name string) bool {
	switch name {
	default:
		return false
	case "Eval", "InputType", "Type", "Slice", "Name", "Unit", "NComp", "Mesh", "SetValue", "String":
		return true
	}
}

// list of examples where entry is used.
func (e *entry) Examples() []int {
	return api_examples[strings.ToLower(e.name)]
}

type api struct {
	Entries entries
}

// include file
func (e *api) Include(fname string) string {
	b, err := os.ReadFile(path.Join(templateDir, fname))
	check(err)
	return string(b)
}

// list of entries not used so far
func (a *api) remaining() []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if !e.touched {
			E = append(E, e)
		}
	}
	return E
}

// list of all entries (touched and not touched)
func (a *api) All() []*entry {
	var E []*entry
	for _, e := range a.Entries {
		E = append(E, e)
	}
	return E
}

// return all entries, unused so far, which have given type.
func (a *api) FilterType(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		for _, t := range typ {
			if match(t, e.Type.String()) &&
				!strings.HasPrefix(e.name, "ext_") {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

// return all entries, unused so far, which have given return type.
func (a *api) FilterReturn(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		for _, t := range typ {
			if match(t, e.Ret()) &&
				!strings.HasPrefix(e.name, "ext_") {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

// return all entries, unused so far, which have given name.
func (a *api) FilterName(typ ...string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		for _, t := range typ {
			if match(t, e.name) &&
				!strings.HasPrefix(e.name, "ext_") {
				e.touched = true
				E = append(E, e)
			}
		}
	}
	return E
}

// return all entries, unused so far, whose name starts with prefix.
func (a *api) FilterPrefix(pre string) []*entry {
	var E []*entry
	for _, e := range a.Entries {
		if strings.HasPrefix(e.name, pre) {
			e.touched = true
			E = append(E, e)
		}
	}
	return E
}

// return all entries not yet used.
func (a *api) FilterLeftovers() []*entry {
	return a.remaining()
}

// case insensitive match.
func match(a, b string) bool {
	a = strings.ToLower(a)
	b = strings.ToLower(b)
	match := a == b
	return match
}

func renderAPI() {
	e := api_entries
	t := template.Must(template.New("api").Parse(templ))
	f, err2 := os.OpenFile(path.Join(buildDir, "api.html"), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)
	check(t.Execute(f, &api{e}))
}

var templ = read(path.Join(templateDir, "api-template.html"))

func read(fname string) string {
	b, err := os.ReadFile(fname)
	check(err)
	return string(b)
}

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
