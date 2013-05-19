package script

type String string

func (s String) String() string {
	return string(s)
}

func (s String) Eval() interface{} {
	return s
}
