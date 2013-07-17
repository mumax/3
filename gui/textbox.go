package gui

import (
	"fmt"
)

// {{.Textbox id value}} adds a textbox to the document.
// value is the initial text in the box.
// optional width?
func (d *Doc) TextBox(id, value string) string {
	e := newElem(id, "value", value)
	d.add(e)
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v onchange="notifytextbox('%v')"/>`, id, id) // todo: onblur...
}
