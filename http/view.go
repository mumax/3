package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"reflect"
	"text/template"
)

type View struct {
	data      reflect.Value
	templ     *template.Template
	haveJS    bool // have called JS()?
	objects   []obj
	htmlCache []byte // static html content, rendered only once
}

func NewView(data interface{}, templ string) *View {
	t := template.Must(template.New("").Parse(templ))
	v := &View{data: reflect.ValueOf(data), templ: t}
	// pre-render static html content
	cache := bytes.NewBuffer(nil)
	check(v.templ.Execute(cache, v))
	v.htmlCache = cache.Bytes()
	if !v.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	fmt.Println("objects:", v.objects)
	return v
}

func (v *View) ListenAndServe(port string) {
	http.HandleFunc("/", v.RenderHTML)
	http.HandleFunc("/refresh/", v.refresh)
	http.HandleFunc("/rpc/", v.rpc)
	check(http.ListenAndServe(":7070", nil))
}

// {{.Static "field"}} renders the value returned by the data's method or field called "field".
// The HTML is static, i.e., will not auto-refresh.
func (v *View) Static(field string) string {
	return htmlEsc(v.call(field))
}

type obj interface {
	Eval() interface{}
}

func (v *View) Label(field string) string {
	id := v.addObj(method(v.data, field))
	return fmt.Sprintf(`<span id=%v>%v</span>`, id, "")
}

func (v *View) rpc(w http.ResponseWriter, r *http.Request) {
	m := make(map[string]string)
	check(json.NewDecoder(r.Body).Decode(&m))
	methodName := m["Method"]
	v.data.MethodByName(methodName).Call([]reflect.Value{})
}

func (v *View) Button(action string) string {
	if v.data.MethodByName(action).Kind() == 0{
		log.Panic("no such method:", action)
	}
	return fmt.Sprintf(`<button onclick="rpc(&quot;%v&quot;);">%v</button>`, action, action)
}

func (m *method_) Eval() interface{} {
	args := []reflect.Value{}
	r := m.Call(args)
	argument(len(r) == 1, "need one return value")
	return r[0].Interface()
}

type method_ struct{ reflect.Value }

func method(v reflect.Value, field string) obj {
	m := v.MethodByName(field)
	if m.Kind() != 0 {
		return &method_{m}
	} else {
		panic(fmt.Sprint("type ", v.Type(), " has no such field or method ", field))
	}
}

func (v *View) addObj(o obj) (id int) {
	v.objects = append(v.objects, o)
	return len(v.objects)
}

// {{.JS}} should be embedded in the template <head>
func (v *View) JS() string {
	v.haveJS = true
	return js
}

// {{.Err}} should be embedded in the template where errors are to be shown.
func (v *View) Err() string {
	return `<span id=Errorbox ></span>`
}

func (v *View) RenderHTML(w http.ResponseWriter, r *http.Request) {
	w.Write(v.htmlCache)
}

// key-value pair
type kv struct{ ID, HTML string }

func (v *View) refresh(w http.ResponseWriter, r *http.Request) {
	fmt.Print("*")
	js := []kv{}
	for i, o := range v.objects {
		id := fmt.Sprint(i + 1)
		innerHTML := htmlEsc(o.Eval())
		js = append(js, kv{id, innerHTML})
	}
	check(json.NewEncoder(w).Encode(js))
}

func (v *View) call(field string) interface{} {
	m := v.data.MethodByName(field)
	if m.Kind() != 0 {
		args := []reflect.Value{}
		r := m.Call(args)
		argument(len(r) == 1, "need one return value")
		return r[0].Interface()
	}
	panic(fmt.Sprint("type ", v.data.Type(), " has no such field or method ", field))
}

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
