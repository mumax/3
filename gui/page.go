package gui

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync"
	"text/template"
)

var Debug = false

// Page holds the state to serve a single GUI page to the browser
type Page struct {
	elems      map[string]*E
	htmlCache  []byte      // static html content, rendered only once
	haveJS     bool        // have called JS()?
	data       interface{} // any additional data to be passed to template
	onUpdate   func()
	onAnyEvent func()
	httpLock   sync.Mutex
	lastPageID string
}

// NewPage constructs a Page based on an HTML template containing
// element tags like {{.Button}}, {{.Textbox}}, etc. data is fed
// to the template as additional arbitrary data, available as {{.Data}}.
func NewPage(htmlTemplate string, data interface{}) *Page {
	d := &Page{elems: make(map[string]*E), data: data}

	// exec template (once)
	t := template.Must(template.New("").Parse(htmlTemplate))
	cache := bytes.NewBuffer(nil)
	check(t.Execute(cache, d))
	d.htmlCache = cache.Bytes()

	// check if template contains {{.JS}}
	if !d.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	return d
}

// Value returns the value of the HTML element with given id.
// E.g.: the text in a textbox, the checked value of a checkbox, etc.
func (d *Page) Value(id string) interface{} {
	return d.elem(id).value()
}

// StringValue is like Value but returns the value as string,
// converting if necessary.
func (d *Page) StringValue(id string) string {
	v := d.Value(id)
	if s, ok := v.(string); ok {
		return s
	} else {
		return fmt.Sprint(v)
	}
}

// BoolValue is like Value but returns the value as bool,
// converting if necessary. Panics if conversion is not possible.
func (d *Page) BoolValue(id string) bool {
	v := d.Value(id)
	if b, ok := v.(bool); ok {
		return b
	} else {
		b, err := strconv.ParseBool(fmt.Sprint(v))
		check(err)
		return b
	}
}

// IntValue is like Value but returns the value as int,
// converting if necessary. Panics if conversion is not possible.
func (d *Page) IntValue(id string) int {
	v := d.Value(id)
	if b, ok := v.(int); ok {
		return b
	} else {
		b, err := strconv.Atoi(fmt.Sprint(v))
		check(err)
		return b
	}
}

// FloatValue is like Value but returns the value as float64,
// converting if necessary. Panics if conversion is not possible.
func (d *Page) FloatValue(id string) float64 {
	v := d.Value(id)
	if b, ok := v.(float64); ok {
		return b
	} else {
		b, err := strconv.ParseFloat(fmt.Sprint(v), 64)
		check(err)
		return b
	}
}

func (d *Page) Set(id string, v interface{}) {
	d.elem(id).set(v)
}

func (d *Page) Attr(id string, k string, v interface{}) {
	d.elem(id).attr(k, v)
}

func (d *Page) Disable(id string, disable bool) {
	d.Attr(id, "disabled", disable)
}

func (d *Page) Display(id string, visible bool) {
	if visible {
		d.Attr(id, "style.display", "none")
	} else {
		d.Attr(id, "style.display", "block")
	}
}

// OnEvent sets a handler to be called when an event happens
// to the HTML element with given id. The event depends on the
// element type: click for Button, change for TextBox, etc...
func (d *Page) OnEvent(id string, handler func()) {
	d.elem(id).onevent = handler
}

// OnEvent sets a handler to be called when an event happens
// to any of the page's HTML elements.
func (d *Page) OnAnyEvent(handler func()) {
	d.onAnyEvent = handler
}

// Set func to be executed each time javascript polls for updates
func (d *Page) OnUpdate(f func()) {
	d.onUpdate = f
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

// {{.UpdateButton}} adds a page Update button
func (t *Page) UpdateButton(text string) string {
	if text == "" {
		text = `&#x21bb;`
	}
	return `<button onclick="update();"> ` + text + ` </button>`
}

// {{.UpdateBox}} adds an auto update checkbox
func (t *Page) UpdateBox(text string) string {
	if text == "" {
		text = "auto update"
	}
	return `<input type=checkbox id=UpdateBox class=CheckBox checked=true onchange="autoUpdate=elementById('UpdateBox').checked").checked">` + text + `</input>`
}

// {{.Data}} returns the extra data that was passed to NewPage
func (t *Page) Data() interface{} {
	return t.data
}

// return elem[id], panic if non-existent
func (d *Page) elem(id string) *E {
	if e, ok := d.elems[id]; ok {
		return e
	} else {
		panic("no element with id: " + id)
	}
}

// elem[id] = e, panic if already defined
func (d *Page) addElem(id string, e El) {
	if _, ok := d.elems[id]; ok {
		panic("addElem: already defined: " + id)
	} else {
		d.elems[id] = newE(e)
	}
}

// ServeHTTP implements http.Handler.
func (d *Page) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	d.httpLock.Lock()
	defer d.httpLock.Unlock()
	switch r.Method {
	default:
		http.Error(w, "not allowed: "+r.Method+" "+r.URL.Path, http.StatusForbidden)
	case "GET":
		d.serveContent(w, r)
	case "POST":
		d.serveUpdate(w, r)
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
	var ev event
	check(json.NewDecoder(r.Body).Decode(&ev))
	if Debug {
		fmt.Println(ev)
	}
	if d.onAnyEvent != nil {
		d.onAnyEvent()
	}
	el := d.elem(ev.ID)
	el.set(ev.Arg)
	if el.onevent != nil {
		el.onevent()
	}
}

type event struct {
	ID  string
	Arg interface{}
}

// HTTP handler for updating the dynamic elements
func (d *Page) serveUpdate(w http.ResponseWriter, r *http.Request) {
	if d.onUpdate != nil {
		d.onUpdate()
	}

	// read page ID from body
	buf := make([]byte, 100)
	r.Body.Read(buf)
	pageID := string(buf)
	if pageID != d.lastPageID {
		for _, e := range d.elems {
			e.setDirty()
		}
		d.lastPageID = pageID
	}

	calls := make([]jsCall, 0, len(d.elems))
	for id, e := range d.elems {
		calls = append(calls, e.update(id)...) // update atomically checks dirty and clears it
	}
	if Debug && len(calls) != 0 {
		fmt.Println(calls) // debug
	}
	check(json.NewEncoder(w).Encode(calls))
}

// javascript call
type jsCall struct {
	F    string        // function to call
	Args []interface{} // function arguments
}

func check(err error) {
	if err != nil {
		log.Panic(err)
	}
}
