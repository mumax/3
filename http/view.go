package main

import (
	"fmt"
	"os"
	"log"
	"net/http"
	"reflect"
	"text/template"
	"encoding/json"
	"bytes"
)

type View struct {
	data   reflect.Value
	templ  *template.Template
	haveJS bool // have called JS()?
	objects []obj
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
	http.HandleFunc("/refresh/", v.Refresh)
	check(http.ListenAndServe(":7070", nil))
}

// {{.Static "field"}} renders the value returned by the data's method or field called "field".
// The HTML is static, i.e., will not auto-refresh.
func (v *View) Static(field string) string {
	return htmlEsc(v.call(field))
}

type obj interface{
	Eval()interface{}
}

func (v *View) Dynamic(field string) string {
	id := v.addObj(method(v.data, field))
	return fmt.Sprintf(`<p id=%v>%v</p>`, id, "")
}


func(m*method_)Eval()interface{}{
		args := []reflect.Value{}
		r := m.Call(args)
		argument(len(r) == 1, "need one return value")
		return r[0].Interface()
}

type method_ struct{ reflect.Value }
func method(v reflect.Value, field string)obj{
	m := v.MethodByName(field)
	if m.Kind() != 0 {
		return &method_{m}
	}else{
	panic(fmt.Sprint("type ", v.Type(), " has no such field or method ", field))
	}
}

func(v*View)addObj(o obj)(id int){
	v.objects = append(v.objects, o)
	return len(v.objects)
}


func (v *View) JS() string {
	v.haveJS = true
	return js
}

func (v *View) RenderHTML(w http.ResponseWriter, r *http.Request) {
	w.Write(v.htmlCache)
}


// key-value pair
type kv struct{ID, HTML string}

func (v *View) Refresh(w http.ResponseWriter, r *http.Request) {
	log.Println("Refresh")

	js := []kv{}
	for i,o := range v.objects{
		id := fmt.Sprint(i)
		innerHTML := htmlEsc(o.Eval())
		js = append(js, kv{id, innerHTML})
	}
	check(json.NewEncoder(os.Stdout).Encode(js))
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
