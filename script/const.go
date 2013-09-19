package script

import "log"
import "reflect"

func Const(e Expr) bool {
	log.Print("Const ", e, " ", reflect.TypeOf(e), ":")
	switch e := e.(type) {
	default:
		log.Println(false)
		return false
	case Conster:
		log.Println(e.Const())
		return e.Const()
	}
}

type Conster interface {
	Const() bool
}
