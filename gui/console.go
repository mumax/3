package gui

import "fmt"

type console struct {
	data
}

func (e *console) update(id string) []jsCall {
	return []jsCall{{F: "setConsoleText", Args: []interface{}{e.value()}}}
}

func (d *Page) Console(id string, rows, cols int, value interface{}, extra ...string) string {
	e := &console{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<textarea id=%v rows=%v cols=%v class=TextBox %v></textarea>`, id, rows, cols, cat(extra))
}
