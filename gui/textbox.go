package gui

import (
	"fmt"
)

// {{.Textbox id value}} adds a textbox to the document.
// value is the initial text in the box.
// optional width?
func (d *Doc) TextBox(id string, value ...string) string {
	val := cat(value)
	e := newElem(id, "value", val)
	d.add(e)
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" onchange="notifytextbox('%v')"/>`, id, val, id) // todo: onblur...
}
