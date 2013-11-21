package gui

import (
	"fmt"
)

// {{.Button id value}} adds a button to the document.
// value is text on the button.
func (t *Templ) Button(id string, value string) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", el.value()}}
	}
	return fmt.Sprintf(`<button id=%v class=Button onclick="notifyButton('%v')">%v</button>`,
		id, id, value) // set button value does not work
}
