package gui

import (
	"sync"
)

// Elem represents a GUI element (button, textbox, ...)
type Elem struct {
	id    string
	value string
	dirty bool
	sync.Mutex
	domAttr string
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

func (e *Elem) SetValue(v string) {
	e.Lock()
	e.dirty = true
	e.value = v
	e.Unlock()
}
