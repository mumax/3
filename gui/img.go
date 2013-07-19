package gui

import "fmt"

// {{.Img id url}} adds an image to the document.
func (d *Doc) Img(id string, value string) string {
	e := newElem(id, "innerHTML", value)
	d.add(e)
	return fmt.Sprintf(`<img id=%v src="%v"/>`, id, value)
}
