package gui

import (
	"sync"
)

type Elem struct {
	id    string
	value string
	dirty bool
	sync.Mutex
}

func newElem(id, value string) *Elem {
	return &Elem{id: id, value: value, dirty: true}
}

func (e *Elem) Id() string {
	return e.id
}

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
