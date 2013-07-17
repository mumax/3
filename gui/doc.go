package gui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"text/template"
)

// gui.Doc serves a GUI as a html document.
type Doc struct {
	haveJS    bool             // have called JS()?
	elem      map[string]*Elem // document elements by ID
	htmlCache []byte           // static html content, rendered only once
	prefix    string           // URL prefix (not yet working)
}

// NewDoc makes a new GUI document, to be served under urlPattern.
// htmlTemplate defines the GUI elements and layout.
// A http handler still needs to be registered manually.
// Example...
func NewDoc(urlPattern, htmlTemplate string) *Doc {
	t := template.Must(template.New(urlPattern).Parse(htmlTemplate))
	d := &Doc{elem: make(map[string]*Elem), prefix: urlPattern}
	cache := bytes.NewBuffer(nil)
	check(t.Execute(cache, d))
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

// Elem returns an element by Id.
func (d *Doc) Elem(id string) *Elem {
	if e, ok := d.elem[id]; ok {
		return e
	} else {
		panic("elem id " + id + " undefined")
	}
}

func (d *Doc) add(e *Elem) {
	id := e.Id()
	if _, ok := d.elem[id]; ok {
		log.Panic("element id " + id + " already defined")
	}
	d.elem[id] = e
}

// ServeHTTP implements http.Handler.
func (d *Doc) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	url := r.URL.Path[len(d.prefix):]
	//log.Println("handle", url)
	switch url {
	default:
		http.Error(w, "not found: "+r.URL.Path, http.StatusNotFound)
	case "":
		d.serveContent(w, r)
	case "refresh/":
		d.serveRefresh(w, r)
	case "rpc":
		d.serveRPC(w, r)
	}
}

// serves the html content.
func (d *Doc) serveContent(w http.ResponseWriter, r *http.Request) {
	for _, e := range d.elem {
		e.dirty = true
	}
	w.Write(d.htmlCache)
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
		if value, dirty := e.Value(); dirty {
			vEsc := htmlEsc(value)
			js = append(js, domUpd{e.Id(), e.domAttr, vEsc})
		}
	}
	check(json.NewEncoder(w).Encode(js))
}

// DOM update action
type domUpd struct {
	ID   string // element ID to update
	ATTR string // element attribute (innerHTML, value, ...)
	HTML string // element value to set
}

func check(e error) {
	if e != nil {
		log.Panic(e)
	}
}

func htmlEsc(s string) string {
	return template.HTMLEscapeString(s)
}
