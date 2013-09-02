package gui

import (
	"fmt"
)

// {{.Textbox id value}} adds a textbox to the document.
// value is the initial text in the box.
func (t *Templ) TextBox(id string, value string) string {
	return t.textbox(id, nil, value)
}

// {{.Numbox id value}} adds a textbox for numbers to the document.
// value is the initial text in the box.
func (t *Templ) NumBox(id string, value float64) string {
	return t.textbox(id, &floatData{interfaceData{0}}, value)
}

func (t *Templ) textbox(id string, dm data, value interface{}) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	if dm != nil {
		el.data = dm
	}
	el.update = func(id string) jsCall {
		return jsCall{F: "setTextbox", Args: []interface{}{id, el.value()}}
	}
	return fmt.Sprintf(`<input type=textbox class=TextBox id=%v value="%v" onchange="notifytextbox('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"/>`, id, value, id, id, id)
}
