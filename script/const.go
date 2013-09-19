package script

func Const(e Expr) bool {
	switch e := e.(type) {
	default:
		return false
	case interface {
		Const() bool
	}:
		return e.Const()
	}
}
