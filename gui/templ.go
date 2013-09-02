package gui

// Provides methods to the html template code passed to NewDoc.
type Templ Doc

// {{.JS}} should always be embedded in the template <head>.
// Expands to needed JavaScript code.
func (t *Templ) JS() string {
	d := (*Doc)(t)
	d.haveJS = true
	return js
}

// {{.ErrorBox}} should be embedded in the template where errors are to be shown.
// CSS rules for class ErrorBox may be set, e.g., to render errors in red.
func (t *Templ) ErrorBox() string {
	return `<span id=ErrorBox class=ErrorBox></span> <span id=MsgBox class=ErrorBox></span>`
}

func (t *Templ) Data() interface{} {
	return (*Doc)(t).data
}
