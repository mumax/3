package gui

import "fmt"

type sel struct {
	data
}

func (e *sel) update(id string) []jsCall {
	return []jsCall{{F: "setSelect", Args: []interface{}{id, e.value()}}}
}

func (d *Page) SelectArray(id string, value string, options []string) string {
	return d.Select(id, value, options...)
}

func (d *Page) Select(id string, value string, options ...string) string {
	e := &sel{data: data{value}}
	d.addElem(id, e)
	html := fmt.Sprintf(`<select id=%v onchange="notifyselect('%v')" onfocus="notifyfocus('%v')" onblur="notifyblur('%v')"> `, id, id, id, id) + "\n"

	for _, o := range options {
		html += fmt.Sprintf(`<option value=%v> %v </option>`, o, o) + "\n"
	}
	html += `</select>`
	return html
}
