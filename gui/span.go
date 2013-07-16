package gui

import (
	"fmt"
)

type Span struct {
	elem
}

func (d *Doc) Span(id, value string) string {
	e := &Span{elem{id, value, true, &d.Mutex}}
	d.add(e)
	return e.Render()
}

func (e *Span) Render() string {
	return fmt.Sprintf(`<span id=%v>%v</span>`, e.Id(), e.Value())
}
