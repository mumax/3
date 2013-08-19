package gui

import "fmt"

func (d *Doc) BeginSelect(id string) string {
	e := newElem(id, "select", 0) // select is dummy attr
	d.add(id, e)
	return fmt.Sprintf(`<select id=%v onchange="notifyselect('%v')"> `, id, id)
}

func (d *Doc) EndSelect() string {
	return `</select>`
}

func (d *Doc) Option(value string) string {
	return fmt.Sprintf(`<option value=%v> %v </option>`, value, value)
}
