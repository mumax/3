package gui

import (
	"sync"
)

type Elem interface {
	Id() string
	Value() (string, bool)
	SetValue(string)
}

type elem struct {
	id    string
	value string
	dirty bool
	sync.Mutex
}

func makeElem(id, value string) elem {
	return elem{id: id, value: value, dirty: true}
}

func (e *elem) Id() string {
	return e.id
}

func (e *elem) Value() (value string, dirty bool) {
	e.Lock()
	value = e.value
	dirty = e.dirty
	e.dirty = false
	e.Unlock()
	return
}

func (e *elem) SetValue(v string) {
	e.Lock()
	e.dirty = true
	e.value = v
	e.Unlock()
}
