package gui

import (
	"fmt"
)

func (t *Templ) BeginSelect(id string) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.update = func(id string) jsCall {
		return jsCall{F: "setSelect", Args: []interface{}{id, el.value()}}
	}
	return fmt.Sprintf(`<select id=%v onchange="notifyselect('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"> `, id, id, id, id)
}

func (t *Templ) EndSelect() string {
	return `</select>`
}

func (t *Templ) Option(value string) string {
	return fmt.Sprintf(`<option value=%v> %v </option>`, value, value)
}
