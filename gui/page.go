package gui

import (
	"bytes"
	"log"
	"sync"
	"text/template"
)

type Page struct {
	elems     map[string]El
	htmlCache []byte      // static html content, rendered only once
	haveJS    bool        // have called JS()?
	data      interface{} // any additional data to be passed to template
	onRefresh func()
	sync.Mutex
}

func NewPage(htmlTemplate string, data interface{}) *Page {
	d := &Page{elems: make(map[string]El),
		data:      data,
		onRefresh: func() {}}

	// exec template (once)
	t := template.Must(template.New("").Parse(htmlTemplate))
	cache := bytes.NewBuffer(nil)
	err := t.Execute(cache, d)
	if err != nil {
		log.Panic(err)
	}
	d.htmlCache = cache.Bytes()

	// check if template contains {{.JS}}
	if !d.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	return d
}

// {{.JS}} should always be embedded in the template <head>.
// Expands to needed JavaScript code.
func (d *Page) JS() string {
	d.haveJS = true
	return JS
}

// {{.ErrorBox}} should be embedded in the template where errors are to be shown.
// CSS rules for class ErrorBox may be set, e.g., to render errors in red.
func (t *Page) ErrorBox() string {
	return `<span id=ErrorBox class=ErrorBox></span> <span id=MsgBox class=ErrorBox></span>`
}

func (t *Page) Data() interface{} {
	return t.data
}
