package gui

import (
	"fmt"
)

type Span struct {
	elem
}

func newSpan(id, value string) *Span {
	return &Span{elem{id, value}}
}

func (e *Span) Render() string {
	return fmt.Sprintf(`<span id=%v>%v</span>`, e.Id(), e.Value())
}
