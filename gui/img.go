package gui

import "fmt"

type img struct {
	data
}

func (e *img) update(id string) []jsCall {
	return []jsCall{{F: "setAttr", Args: []interface{}{id, "src", e.value()}}}
}

func (d *Page) Img(id string, value interface{}, extra ...string) string {
	e := &img{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<img id=%v %v/> `, id, cat(extra))
}
