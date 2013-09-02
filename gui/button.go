package gui

import (
	"fmt"
)

// {{.Button id value}} adds a button to the document.
// value is text on the button.
func (t *Templ) Button(id string, value ...string) string {
	val := id
	if len(value) != 0 {
		val = cat(value)
	}
	d := (*Doc)(t)
	el := d.addElem(id)
	el.setValue(val)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "innerHTML", el.value()}}
	}
	return fmt.Sprintf(`<button id=%v class=Button onclick="notifyButton('%v')">%v</button>`,
		id, id, val) // set button value does not work
}

func cat(value []string) string {
	if len(value) == 0 {
		return ""
	} else {
		cat := value[0]
		for i := 1; i < len(value); i++ {
			cat += " " + value[i]
		}
		return cat
	}
}
