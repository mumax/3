package gui

import (
	"fmt"
)

func (d *Doc) Span(id, value string) string {
	e := newElem(id, value)
	d.add(e)
	return fmt.Sprintf(`<span id=%v>%v</span>`, e.Id(), "")
}
