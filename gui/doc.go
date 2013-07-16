package gui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"text/template"
)

type Doc struct {
	templ     *template.Template
	haveJS    bool // have called JS()?
	elem      map[string]Elem
	htmlCache []byte // static html content, rendered only once
	prefix    string
	sync.Mutex
}

func NewDoc(urlPattern, htmlTemplate string) *Doc {
	t := template.Must(template.New(urlPattern).Parse(htmlTemplate))
	d := &Doc{templ: t, elem: make(map[string]Elem), prefix: urlPattern}
	cache := bytes.NewBuffer(nil)
	check(d.templ.Execute(cache, d))
	if !d.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	d.htmlCache = cache.Bytes()
	http.Handle(urlPattern, d)
	return d
}

// {{.JS}} should always be embedded in the template <head>.
// Expands to needed JavaScript code.
func (d *Doc) JS() string {
	d.haveJS = true
	return js
}

// {{.ErrorBox}} should be embedded in the template where errors are to be shown.
// CSS rules for class ErrorBox may be set, e.g., to render errors in red.
func (d *Doc) ErrorBox() string {
	return `<span id=ErrorBox class=ErrorBox></span> <span id=MsgBox class=ErrorBox></span>`
}

// {{.AutoRefreshBox }} renders a check box to toggle auto-refresh.
func (v *Doc) AutoRefreshBox() string {
	return fmt.Sprintf(`<input type="checkbox" id="AutoRefresh" checked=true onchange="setautorefresh()">auto refresh</input>`)
}

//{{.TextBox}}
//func (v *Doc) TextBox(modelName string) string {
//	_ = v.getModel(modelName) // check existence
//	id := id(modelName)
//	i := "guielem_" + modelName
//	return fmt.Sprintf(`<input type=text id=%v class=TextBox onchange="settext('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, modelName, i, i)
//}

func (d *Doc) Elem(id string) Elem {
	if e, ok := d.elem[id]; ok {
		return e
	} else {
		panic("elem id " + id + " undefined")
	}
}

func (d *Doc) add(e Elem) {
	id := e.Id()
	if _, ok := d.elem[id]; ok {
		log.Panic("element id " + id + " already defined")
	}
	d.elem[id] = e
}

func (d *Doc) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len(d.prefix):]
	log.Println("handle", url)
	switch url {
	default:
		http.Error(w, "not found: "+r.URL.Path, http.StatusNotFound)
	case "":
		w.Write(d.htmlCache)
	case "refresh/":
		d.serveRefresh(w, r)
	case "rpc":
		d.serveRPC(w, r)
	}
}

// HTTP handler for RPC calls by button clicks etc
func (v *Doc) serveRPC(w http.ResponseWriter, r *http.Request) {
	m := make(map[string]string)
	check(json.NewDecoder(r.Body).Decode(&m))
	log.Println("RPC", m)
	//	modelName := m["ID"]
	//	method := m["Method"]
	//	switch method {
	//	default:
	//		panic("rpc: unhandled method: " + method)
	//	case "call":
	//		v.caller(modelName).Call()
	//	case "set":
	//		v.setter(modelName).Set(m["Arg"])
	//	}
}

// HTTP handler for refreshing the dynamic elements
func (v *Doc) serveRefresh(w http.ResponseWriter, r *http.Request) {
	//fmt.Print("*")
	js := []domUpd{}
	for _, e := range v.elem {
		if e.Dirty() {
			value := template.HTMLEscapeString(e.Value())
			js = append(js, domUpd{e.Id(), value})
		}
	}
	check(json.NewEncoder(w).Encode(js))
}

// DOM update action
type domUpd struct {
	ID   string // element ID to update
	HTML string // element value to set
}

func check(e error) {
	if e != nil {
		log.Panic(e)
	}
}
