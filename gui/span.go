package gui

import "fmt"

// {{.Span id value}} adds a piece of text ("label") to the document.
func (d *Doc) Span(id string, value ...string) string {
	val := cat(value)
	e := newElem(id, "innerHTML", val)
	d.add(id, e)
	return fmt.Sprintf(`<span id=%v>%v</span>`, id, val)
}

func cat(value []string) string {
	val := ""
	for _, v := range value {
		val += v + " "
	}
	return val
}
