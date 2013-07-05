package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"reflect"
	"text/template"
)

type View struct {
	data   reflect.Value
	templ  *template.Template
	haveJS bool // have called JS()?
}

func NewView(data interface{}, templ string) *View {
	t := template.Must(template.New("").Parse(templ))
	v := &View{data: reflect.ValueOf(data), templ: t}

	v.Render(ioutil.Discard) // test run
	if !v.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	return v
}

// {{.Static "field"}} renders the value returned by the data's method or field called "field".
// The HTML is static, i.e., will not auto-refresh.
func (v *View) Static(field string) string {
	return htmlEsc(v.call(field))
}

//func(v*View)Dynamic(field string)string{
//
//}

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

func (v *View) JS() string {
	v.haveJS = true
	return js
}

func (v *View) Render(out io.Writer) {
	check(v.templ.Execute(out, v))
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
