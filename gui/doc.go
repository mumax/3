package gui

import (
	"bytes"
	"log"
	"text/template"
	"time"
)

// gui.Doc serves a GUI as a html document.
type Doc struct {
	elems     map[string]*elem
	htmlCache []byte      // static html content, rendered only once
	haveJS    bool        // have called JS()?
	data      interface{} // any additional data to be passed to template
	keepAlive time.Time   // last time we heard from the browser
	onRefresh func()
}

func NewDoc(htmlTemplate string, data interface{}) *Doc {
	d := &Doc{elems: make(map[string]*elem), data: data, onRefresh: func() {}}
	d.execTemplate(htmlTemplate)
	if !d.haveJS {
		log.Panic("template should call {{.JS}}")
	}
	return d
}

func (d *Doc) OnEvent(id string, f func()) {
	d.elem(id).onevent = f
}

func (d *Doc) OnRefresh(f func()) {
	d.onRefresh = f
}

func (d *Doc) SetValue(id string, v interface{}) {
	d.elem(id).setValue(v)
}

func (d *Doc) Value(id string) interface{} {
	return d.elem(id).value()
}

func (d *Doc) addElem(id string) *elem {
	if _, ok := d.elems[id]; ok {
		panic("doc.addElem: already defined: " + id)
	} else {
		el := &elem{data: &interfaceData{nil}}
		d.elems[id] = el
		return el
	}
}

func (d *Doc) elem(id string) *elem {
	if e, ok := d.elems[id]; ok {
		return e
	} else {
		panic("no element with id: " + id)
	}
}

// parse and execute template, store result in d.htmlCache.
func (d *Doc) execTemplate(htmlTemplate string) {
	t := template.Must(template.New("").Parse(htmlTemplate))
	cache := bytes.NewBuffer(nil)
	check(t.Execute(cache, (*Templ)(d)))
	d.htmlCache = cache.Bytes()
}

func check(e error) {
	if e != nil {
		log.Panic(e)
	}
}
