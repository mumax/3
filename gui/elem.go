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
	*sync.Mutex
}

func (e *elem) Id() string {
	return e.id
}

func (e *elem) Value() string {
	return e.value
}

func (e *elem) SetValue(v string) {
	e.value = v
}

func (e *elem) Dirty() bool {
	return e.dirty
}
