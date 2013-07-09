package gui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	//"reflect"
	"text/template"
)

type Server struct {
	templ     *template.Template
	haveJS    bool // have called JS()?
	model     map[string]interface{}
	htmlCache []byte // static html content, rendered only once
}

func NewServer(templ string) *Server {
	t := template.Must(template.New("").Parse(templ))
	return &Server{templ: t, model: make(map[string]interface{})}
}

// pre-render static html content
func (v *Server) preRender() {
	cache := bytes.NewBuffer(nil)
	check(v.templ.Execute(cache, v))
	if !v.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	v.htmlCache = cache.Bytes()
}

func (v *Server) ListenAndServe(port string) {
	v.preRender()
	http.HandleFunc("/", v.renderHTML)
	http.HandleFunc("/refresh/", v.refresh)
	http.HandleFunc("/rpc/", v.rpc)
	check(http.ListenAndServe(":7070", nil))
}

// {{.JS}} should always be embedded in the template <head>.
// Expands to needed JavaScript code.
func (v *Server) JS() string {
	v.haveJS = true
	return js
}

// {{.ErrorBox}} should be embedded in the template where errors are to be shown.
// CSS rules for class ErrorBox may be set, e.g., to render errors in red.
func (v *Server) ErrorBox() string {
	return `<span id=ErrorBox class=ErrorBox></span> <span id=MsgBox class=ErrorBox></span>`
}

//// {{.Static "modelName"}}
//// without auto-refresh.
//func (v *Server) Static(method string) string {
//	return htmlEsc(v.call(method))
//}

// {{.Label "modelName"}}
// with auto-refresh.
func (v *Server) Label(modelName string) string {
	m := v.getter(modelName)
	return fmt.Sprintf(`<span id=%v>%v</span>`, id(modelName), m.Get())
}

// {{.Button "modelName"}}
func (v *Server) Button(modelName string) string {
	_ = v.caller(modelName) // check existence
	id := id(modelName)
	return fmt.Sprintf(`<button id=%v onclick="call('%v')">%v</button>`, id, modelName, modelName)
}

// {{.AutoRefreshBox }} renders a check box to toggle auto-refresh.
func (v *Server) AutoRefreshBox() string {
	return fmt.Sprintf(`<input type="checkbox" id="AutoRefresh" checked=true onchange="setautorefresh()">auto refresh</input>`)
}

//{{.TextBox}}
func (v *Server) TextBox(modelName string) string {
	_ = v.getModel(modelName) // check existence
	id := id(modelName)
	i := "guielem_" + modelName
	return fmt.Sprintf(`<input type=text id=%v class=TextBox onchange="settext('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, modelName, i, i)
}

func id(name string) string {
	return `"guielem_` + name + `"`
}

func (v *Server) getter(name string) Getter {
	m := v.getModel(name)
	if g, ok := m.(Getter); ok {
		return g
	} else {
		panic("model " + name + " has no get() functionality")
	}
}

func (v *Server) setter(name string) Setter {
	m := v.getModel(name)
	if s, ok := m.(Setter); ok {
		return s
	} else {
		panic("model " + name + " has no set() functionality")
	}
}

func (v *Server) caller(name string) Caller {
	m := v.getModel(name)
	if s, ok := m.(Caller); ok {
		return s
	} else {
		panic("model " + name + " has no call() functionality")
	}
}

func (v *Server) getModel(name string) interface{} {
	if m, ok := v.model[name]; ok {
		return m
	} else {
		panic("undefined model: " + name)
	}
}

// HTTP handler for the main page
func (v *Server) renderHTML(w http.ResponseWriter, r *http.Request) {
	w.Write(v.htmlCache)
}

// HTTP handler for refreshing the dynamic elements
func (v *Server) refresh(w http.ResponseWriter, r *http.Request) {
	//fmt.Print("*")
	js := []domUpd{}
	for n, m := range v.model {
		if g, ok := m.(Getter); ok {
			id := "guielem_" + n // no quotes!
			innerHTML := htmlEsc(g.Get())
			js = append(js, domUpd{id, innerHTML})
		}
	}
	check(json.NewEncoder(w).Encode(js))
}

// DOM update action
type domUpd struct {
	ID   string // element ID to update
	HTML string // element value to set
}

// HTTP handler for RPC calls by button clicks etc
func (v *Server) rpc(w http.ResponseWriter, r *http.Request) {
	m := make(map[string]string)
	check(json.NewDecoder(r.Body).Decode(&m))
	log.Println("RPC", m)
	modelName := m["ID"]
	method := m["Method"]
	switch method {
	default:
		panic("rpc: unhandled method: " + method)
	case "call":
		v.caller(modelName).Call()
	case "set":
		v.setter(modelName).Set(m["Arg"])
	}
}

//func (v *Server) call(field string) string {
//	//m := v.data.MethodByName(field)
//	//if m.Kind() != 0 {
//	//	r := m.Call([]reflect.Value{})
//	//	argument(len(r) == 1, "need one return value")
//	//	return fmt.Sprint(r[0].Interface())
//	//}
//	//panic(fmt.Sprint("type ", v.data.Type(), " has no such field or method ", field))
//}

func htmlEsc(v interface{}) string {
	return template.HTMLEscapeString(fmt.Sprint(v))
}

func check(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func argument(test bool, msg string) {
	if !test {
		log.Panic(msg)
	}
}
