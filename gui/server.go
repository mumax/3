package gui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"reflect"
	"text/template"
)

type Server struct {
	data      reflect.Value
	templ     *template.Template
	haveJS    bool // have called JS()?
	elements  []fmt.Stringer
	htmlCache []byte // static html content, rendered only once
}

func NewServer(data interface{}, templ string) *Server {
	t := template.Must(template.New("").Parse(templ))
	v := &Server{data: reflect.ValueOf(data), templ: t}

	// pre-render static html content
	cache := bytes.NewBuffer(nil)
	check(v.templ.Execute(cache, v))
	if !v.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	v.htmlCache = cache.Bytes()
	return v
}

func (v *Server) ListenAndServe(port string) {
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

// {{.Static "method"}} renders the value returned by the data's specified method,
// without auto-refresh.
func (v *Server) Static(method string) string {
	return htmlEsc(v.call(method))
}

// {{.Label "method"}} renders the value returned by the data's specified method,
// with auto-refresh.
func (v *Server) Label(meth string) string {
	id := v.addElem(method(v.data, meth))
	return fmt.Sprintf(`<span id=%v>%v</span>`, id, "")
}

// {{.Button "method"}} renders a button that invokes the specified method on click.
func (v *Server) Button(method string) string {
	if v.data.MethodByName(method).Kind() == 0 {
		log.Panic("no such method:", method)
	}
	return fmt.Sprintf(`<button onclick="rpc(&quot;%v&quot;);">%v</button>`, method, method)
}

func (m *method_) String() string {
	args := []reflect.Value{}
	r := m.Call(args)
	argument(len(r) == 1, "need one return value")
	return fmt.Sprint(r[0].Interface())
}

type method_ struct{ reflect.Value }

func method(v reflect.Value, field string) fmt.Stringer {
	m := v.MethodByName(field)
	if m.Kind() != 0 {
		return &method_{m}
	} else {
		panic(fmt.Sprint("type ", v.Type(), " has no such field or method ", field))
	}
}

func (v *Server) addElem(o fmt.Stringer) (id int) {
	v.elements = append(v.elements, o)
	return len(v.elements)
}

// HTTP handler for the main page
func (v *Server) renderHTML(w http.ResponseWriter, r *http.Request) {
	w.Write(v.htmlCache)
}

// HTTP handler for refreshing the dynamic elements
func (v *Server) refresh(w http.ResponseWriter, r *http.Request) {
	//fmt.Print("*")
	js := []kv{}
	for i, o := range v.elements {
		id := fmt.Sprint(i + 1)
		innerHTML := htmlEsc(o.String())
		js = append(js, kv{id, innerHTML})
	}
	check(json.NewEncoder(w).Encode(js))
}

// key-value pair
type kv struct{ ID, HTML string }

// HTTP handler for RPC calls by button clicks etc
func (v *Server) rpc(w http.ResponseWriter, r *http.Request) {
	m := make(map[string]string)
	check(json.NewDecoder(r.Body).Decode(&m))
	methodName := m["Method"]
	v.data.MethodByName(methodName).Call([]reflect.Value{})
}

func (v *Server) call(field string) string {
	m := v.data.MethodByName(field)
	if m.Kind() != 0 {
		r := m.Call([]reflect.Value{})
		argument(len(r) == 1, "need one return value")
		return fmt.Sprint(r[0].Interface())
	}
	panic(fmt.Sprint("type ", v.data.Type(), " has no such field or method ", field))
}

func htmlEsc(v string) string {
	return template.HTMLEscapeString(v)
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
