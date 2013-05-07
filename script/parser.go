package script

import (
	"fmt"
	"strconv"
)

// TODO rm
func log(msg ...interface{}) {
	fmt.Println("--", fmt.Sprint(msg...))
}

// node of the parse tree
type fn func() interface{}

func parseLine(l *lexer) fn {
	log("inside line")
	l.advance()
	if l.typ == EOF {
		return nil // marks end of input
	}
	if l.typ == EOL {
		return nop
	}
	if l.typ == IDENT {
		fn := afterIdent(l)
		l.advance()
		if l.typ == EOL || l.typ == EOF {
			return fn
		}
	}
	return l.unexpected()
}

func afterIdent(l *lexer) fn {
	log("after ident")
	ident := l.str
	l.advance()
	switch l.typ {
	default:
		return l.unexpected()
	case LPAREN:
		return insideCall(l, ident)
	case ASSIGN:
		return insideAssign(l, ident)
	}
}

func insideCall(l *lexer, ident string) fn {
	log("inside call")
	args, err := insideArgs(l)
	if err != nil {
		return err
	} else {
		return func() interface{} { return eval(ident, args) }
	}
}

func insideAssign(l *lexer, ident string) fn {
	log("inside assign")
	return nil
}

func afterNum(l *lexer) fn {
	log("inside num")
	val, err := strconv.ParseFloat(l.str, 64)
	if err != nil {
		panic(err)
	}
	return func() interface{} { return val }
}

func insideArgs(l *lexer) (args []fn, err fn) {
	log("inside args")
	l.advance()
	for {
		switch l.typ {
		case RPAREN:
			return args, nil
		case NUM:
			args = append(args, afterNum(l))
		case IDENT:
			args = append(args, afterIdent(l))
		default:
			return nil, l.unexpected()
		}
		l.advance()
		if l.typ == COMMA {
			l.advance()
			if l.typ == RPAREN {
				return nil, l.unexpected()
			}
		}
	}
}
