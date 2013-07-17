package gui

import (
	"fmt"
)

// {{.CheckBox id value}} adds a checkbox to the document.
// value is "true" or "false"
func (d *Doc) CheckBox(id, text string, value bool) string {
	e := newElem(id, "checked", value)
	d.add(e)
	return fmt.Sprintf(`<input type=checkbox id=%v class=CheckBox onchange="notifycheckbox('%v')">%v</input>`, id, id, text)
}
