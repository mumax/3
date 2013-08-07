package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/engine"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strings"
	"text/template"
	"unicode"
)

func main() {

	cuda.Init()
	cuda.LockThread()

	ident := engine.World.Identifiers
	doc := engine.World.Doc
	e := make(entries, 0, len(ident))
	for k, v := range doc {
		t := ident[strings.ToLower(k)].Type()
		e = append(e, entry{k, t, v})
	}
	sort.Sort(&e)

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
	nm := e.Type.NumMethod()
	m := make([]string, 0, nm)
	for i := 0; i < nm; i++ {
		n := e.Type.Method(i).Name
		if unicode.IsUpper(rune(n[0])) {
			m = append(m, n)
		}
	}
	return m
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

func check(err error) {
	if err != nil {
		panic(err)
	}
}

const templ = `
{{.Include "head.html"}}

<body>

<h1> mx3 API </h1>
<hr/>

{{range .Entries}}

<span style="color:#000088; font-size:1.3em">{{.Name}}</span> &nbsp; &nbsp; <span style="color:gray; font-size:0.9em"> {{.Type}}  </span> <br/>

{{with .Doc}} <p> {{.}} </p> {{end}}

{{with .Methods}} <p> <span style="color:grey"> <b>methods:</b> {{range .}} {{.}} {{end}} </span> </p> {{end}}

<br/><hr/>

{{end}}

</body>
`
