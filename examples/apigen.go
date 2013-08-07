package main

import (
	"code.google.com/p/mx3/cuda"
	_ "code.google.com/p/mx3/ext"
	"code.google.com/p/mx3/engine"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strings"
	"text/template"
	"unicode"
)

var (
	api_entries entries
	api_ident   = make(map[string]entry)
)

func buildAPI() {
	cuda.Init()
	cuda.LockThread()

	ident := engine.World.Identifiers
	doc := engine.World.Doc
	e := make(entries, 0, len(ident))
	for K, v := range doc {
		k := strings.ToLower(K)
		t := ident[k].Type()
		entr := entry{K, t, v}
		e = append(e, entr)
		api_ident[k] = entr
	}
	sort.Sort(&e)
	api_entries = e
}

func renderAPI() {
	e := api_entries
	t := template.Must(template.New("api").Parse(templ))
	f, err2 := os.OpenFile("api.html", os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	check(err2)
	check(t.Execute(f, &api{e}))
}

type entry struct {
	Name string
	Type reflect.Type
	Doc  string
}

type entries []entry

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
		if unicode.IsUpper(rune(n[0])) {
			m = append(m, n)
		}
	}
	return m
}

func (e *entry) Examples() []int {
	return api_examples[strings.ToLower(e.Name)]
}

func (e *entries) Len() int {
	return len(*e)
}

func (e *entries) Less(i, j int) bool {
	return strings.ToLower((*e)[i].Name) < strings.ToLower((*e)[j].Name)
}

func (e *entries) Swap(i, j int) {
	(*e)[i], (*e)[j] = (*e)[j], (*e)[i]
}

type api struct {
	Entries entries
}

func (e *api) Include(fname string) string {
	b, err := ioutil.ReadFile(fname)
	check(err)
	return string(b)
}

const templ = `
{{.Include "head.html"}}

<body>

<h1> mx3 API </h1>
<hr/>

{{range .Entries}}

<span style="color:#000088; font-size:1.3em">{{.Name}}</span> &nbsp; &nbsp; <span style="color:gray; font-size:0.9em"> {{.Type}}  </span> <br/>

{{with .Doc}} <p> {{.}} </p> {{end}}

{{with .Examples}} <p> <b>examples:</b> {{range .}} <a href="examples.html#example{{.}}">[{{.}}]</a> {{end}} </p> {{end}}
{{with .Methods}} <p> <span style="color:grey"> <b>methods:</b> {{range .}} {{.}} {{end}} </span> </p> {{end}}

<br/><hr/>

{{end}}

</body>
`
