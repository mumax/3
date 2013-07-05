package main

import (
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
