package gui

import (
	"fmt"
)

// {{.CheckBox id text value}} adds a checkbox to the document.
// text is displayed next to the textbox.
// value is true (checked) or false (unchecked)
func (d *Doc) CheckBox(id, text string, value bool) string {
	e := newElem(id, "checked", value)
	d.add(e)
	return fmt.Sprintf(`<input type=checkbox id=%v class=CheckBox onchange="notifycheckbox('%v')">%v</input>`, id, id, text)
}
