package gui

import (
	"fmt"
	"sync"
)

// Elem represents a GUI element (button, textbox, ...)
type Elem struct {
	id      string
	value   string
	domAttr string // element attribute to assign value to (e.g., "innerHTML")
	dirty   bool   // value needs to be sent on next refresh?
	sync.Mutex
	onclick func() // event handler for clicks
}

func newElem(id, attr, value string) *Elem {
	return &Elem{id: id, value: value, dirty: true, domAttr: attr}
}

func (e *Elem) Id() string {
	return e.id
}

// Value returns the GUI element's value.
// E.g., a textbox's text.
func (e *Elem) Value() (value string, dirty bool) {
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
	e.value = fmt.Sprint(v)
	e.Unlock()
}

func (e *Elem) OnClick(f func()) {
	e.onclick = f
}
