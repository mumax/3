package gui

import (
	"fmt"
)

// {{.Textbox id value}} adds a textbox to the document.
// value is the initial text in the box.
func (t *Templ) TextBox(id string, value string) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setTextbox", Args: []interface{}{id, el.value()}}
	}
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, value, id, id, id)
}
