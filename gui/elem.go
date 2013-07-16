package gui

type Elem interface {
	Id() string
	Value() string
	Dirty() bool
}

type elem struct {
	id    string
	value string
	dirty bool
}

func (e *elem) Id() string    { return e.id }
func (e *elem) Value() string { return e.value }
func (e *elem) Dirty() bool   { return e.dirty }
