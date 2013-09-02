package gui

import (
	"fmt"
)

// {{.CheckBox id text value}} adds a checkbox to the document.
// text is displayed next to the textbox.
// value is true (checked) or false (unchecked)
func (t *Templ) CheckBox(id, text string, value bool) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.data = &boolData{interfaceData{nil}}
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "checked", el.value().(bool)}}
	}
	return fmt.Sprintf(`<input type=checkbox id=%v class=CheckBox onchange="notifycheckbox('%v')">%v</input>`, id, id, text)
}
