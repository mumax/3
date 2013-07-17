package gui

import (
	"sync"
)

// Elem represents a GUI element (button, textbox, ...)
type Elem struct {
	id      string
	value   interface{}
	domAttr string // element attribute to assign value to (e.g., "innerHTML")
	dirty   bool   // value needs to be sent on next refresh?
	sync.Mutex
	onclick, onchange func() // event handler
}

func newElem(id, attr string, value interface{}) *Elem {
	return &Elem{id: id, value: value, dirty: true, domAttr: attr}
}

func (e *Elem) Id() string {
	return e.id
}

// Value returns the GUI element's value.
// E.g., a textbox's text.
func (e *Elem) Value() interface{} {
	e.Lock()
	defer e.Unlock()
	return e.value
}

// returns the value ad whether it is dirty (needs refresh),
// then sets dirty to false.
func (e *Elem) valueDirty() (value interface{}, dirty bool) {
	e.Lock()
	value = e.value
	dirty = e.dirty
	e.dirty = false
	e.Unlock()
	return
}

func (e *Elem) SetValue(v interface{}) {
	e.Lock()
	e.dirty = true
	e.value = v
	e.Unlock()
}

func (e *Elem) OnClick(f func()) {
	e.onclick = f
}

func (e *Elem) OnChange(f func()) {
	e.onchange = f
}
