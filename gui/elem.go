package gui

import (
	"sync"
)

type Elem interface {
	Id() string
	Value() string
	SetValue(string)
	Dirty() bool
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

func (e *elem) Value() string {
	e.dirty = false
	return e.value
}

func (e *elem) SetValue(v string) {
	e.dirty = true
	e.value = v
}

func (e *elem) Dirty() bool {
	return e.dirty
}
