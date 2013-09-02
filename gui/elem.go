package gui

type elem struct {
	data
	onevent func()
	update  func(id string) jsCall
}

type data interface {
	setValue(v interface{})
	value() interface{}
}
