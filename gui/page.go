package gui

import (
	"bytes"
	"log"
	"net/http"
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

// {{.Data}} returns the extra data that was passed to NewPage
func (t *Page) Data() interface{} {
	return t.data
}

// return elem[id], panic if non-existent
func (d *Page) elem(id string) El {
	if e, ok := d.elems[id]; ok {
		return e
	} else {
		panic("no element with id: " + id)
	}
}

// ServeHTTP implements http.Handler.
func (d *Page) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	default:
		http.Error(w, "not allowed: "+r.Method+" "+r.URL.Path, http.StatusForbidden)
	case "GET":
		d.serveContent(w, r)
	case "POST":
		d.serveRefresh(w, r)
	case "PUT":
		d.serveEvent(w, r)
	}
}

// serves the html content.
func (d *Page) serveContent(w http.ResponseWriter, r *http.Request) {
	w.Write(d.htmlCache)
}

// HTTP handler for event notifications by button clicks etc
func (d *Page) serveEvent(w http.ResponseWriter, r *http.Request) {
	//var ev event
	//check(json.NewDecoder(r.Body).Decode(&ev))
	//el := d.elem(ev.ID)
	//el.setValue(ev.Arg)
	//if el.onevent != nil {
	//	el.onevent()
	//}
}

type event struct {
	ID  string
	Arg interface{}
}

// HTTP handler for refreshing the dynamic elements
func (d *Page) serveRefresh(w http.ResponseWriter, r *http.Request) {
	//d.onRefresh()
	//calls := make([]jsCall, 0, len(d.elems))
	//for id, el := range d.elems {
	//	calls = append(calls, el.update(id))
	//}
	//check(json.NewEncoder(w).Encode(calls))
}

// javascript call
type jsCall struct {
	F    string        // function to call
	Args []interface{} // function arguments
}
