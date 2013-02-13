package data

import "log"

func argument(test bool) {
	if test == false {
		log.Panic("illegal argument")
	}
}
