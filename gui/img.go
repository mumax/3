package gui

import "fmt"

// {{.Img id url}} adds an image to the document.
func (d *Doc) Img(id string, value string) string {
	e := newElem(id, "src", value)
	d.add(id, e)
	return fmt.Sprintf(`<img id=%v src="%v"/>`, id, value)
}
