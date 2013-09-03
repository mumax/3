package gui

import "fmt"

// {{.Img id url}} adds an image to the document.
func (t *Templ) Img(id string, value string) string {
	d := (*Doc)(t)
	el := d.addElem(id)
	el.setValue(value)
	el.update = func(id string) jsCall {
		return jsCall{F: "setAttr", Args: []interface{}{id, "src", el.value()}}
	}
	return fmt.Sprintf(`<img id=%v src="%v"/>`, id, value)
}
