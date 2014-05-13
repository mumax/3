package gui

import "fmt"

type clibox struct {
	data
}

func (e *clibox) update(id string) []jsCall {
	return []jsCall{} // We never set the value of the CLI box, only the user does
}

// Command-line interface textbox where user types commands.
func (d *Page) CliBox(id string, value interface{}, extra ...string) string {
	e := &clibox{data: data{value}}
	d.addElem(id, e)
	return fmt.Sprintf(`<input type=%v class=TextBox id=%v  %v />`, "text", id, cat(extra))
}
