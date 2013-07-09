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
	model     map[string]*mod
	htmlCache []byte // static html content, rendered only once
}

func NewServer(templ string) *Server {
	t := template.Must(template.New("").Parse(templ))
	return &Server{templ: t, model: make(map[string]*mod)}
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
	return `<span id=ErrorBox class=ErrorBox></span>`
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

func id(name string) string {
	return "guielem_" + name
}

func (v *Server) getter(name string) Getter {
	m := v.getModel(name)
	if m.get == nil {
		panic("model " + name + " has no get() functionality")
	}
	return m
}

func (v *Server) getModel(name string) *mod {
	if m, ok := v.model[name]; ok {
		return m
	} else {
		panic("undefined model: " + name)
	}
}

//// {{.Button "modelName"}}
//func (v *Server) Button(method string) string {
//	if v.data.MethodByName(method).Kind() == 0 {
//		log.Panic("no such method:", method)
//	}
//	return fmt.Sprintf(`<button onclick="rpc(&quot;%v&quot;);">%v</button>`, method, method)
//}
//
//// {{.AutoRefreshBox }} renders a check box to toggle auto-refresh.
//func (v *Server) AutoRefreshBox() string {
//	return fmt.Sprintf(`<input type="checkbox" id="AutoRefresh" checked=true onchange="setautorefresh();">auto refresh</input>`)
//}
//
//func (v *Server) TextBox(meth string) string {
//	id := v.addElem(method(v.data, meth))
//	return fmt.Sprintf(`<input id=%v class=TextBox onchange="rpc(&quot;%v&quot;, document.getElementById(&quot;%v&quot;).text);"></input>`, id, id, meth)
//}

// HTTP handler for the main page
func (v *Server) renderHTML(w http.ResponseWriter, r *http.Request) {
	w.Write(v.htmlCache)
}

// HTTP handler for refreshing the dynamic elements
func (v *Server) refresh(w http.ResponseWriter, r *http.Request) {
	//fmt.Print("*")
	js := []domUpd{}
	for n, m := range v.model {
		if m.get != nil {
			innerHTML := htmlEsc(m.get())
			js = append(js, domUpd{id(n), "innerHTML", innerHTML})
		}
	}
	check(json.NewEncoder(w).Encode(js))
}

// DOM update action
type domUpd struct {
	ID   string // element ID to update
	Var  string // element member, e.g. innerHTML
	HTML string // element value to set
}

// HTTP handler for RPC calls by button clicks etc
func (v *Server) rpc(w http.ResponseWriter, r *http.Request) {
	//m := make(map[string]string)
	//check(json.NewDecoder(r.Body).Decode(&m))
	//log.Println("RPC", m)

	//methodName := m["Method"]
	//v.data.MethodByName(methodName).Call([]reflect.Value{})
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
