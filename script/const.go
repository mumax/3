package script

func Cnst(e Expr) bool { // TODO: rm (unused)
	switch e := e.(type) {
	default:
		return false
	case interface {
		Cnst() bool
	}:
		return e.Cnst()
	}
}
