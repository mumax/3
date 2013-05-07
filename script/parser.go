package script

import (
	"fmt"
	"strconv"
)

// TODO rm
func enter(msg ...interface{}) {
	fmt.Println(">>", fmt.Sprint(msg...))
}
func exit(msg...interface{}){
	fmt.Println("<<", fmt.Sprint(msg...))
}

// node of the parse tree
type fn func() interface{}

func parseLine(l *lexer) fn {
	enter("line")
	defer exit("line")

	l.advance()
	switch l.typ {
	case EOF:
		return nil // marks end of input
	case EOL:
		return nop // empty line
	default:
		fn := parseExpr(l)
		l.advance()
		if l.typ == EOL || l.typ == EOF { // statement has to be terminated
			return fn
		}else{
		return l.unexpected()
		}
	}
}

func parseIdent(l *lexer) fn {
	enter("ident")
	defer exit("ident")
	switch l.peekTyp {
	case LPAREN:
		return parseCall(l)
	case ASSIGN:
		return parseAssign(l)
	default:
		return makeVariable(l.str) 
	}
}

func parseExpr(l*lexer)fn{
	enter("expr")
	defer exit("expr")

	switch l.typ{
		case IDENT: return parseIdent(l)
		case NUM: return parseNum(l)
		default: return l.unexpected()
		// TODO: handle parens, commas
	}
}


func parseCall(l *lexer) fn {
	enter("call")
	defer exit("call")
	funcname := l.str
	l.advance()
	assert(l.typ == LPAREN)
	args, err := parseArgs(l)
	if err != nil {
		return err
	} else {
		return func() interface{} { return call(funcname, args) }
	}
}

func parseAssign(l *lexer) fn {
	enter("assign")
	defer exit("assign")

	left := l.str
	l.advance()
	assert(l.typ == ASSIGN)
	l.advance()
	right := parseExpr(l)
	return func()interface{}{ fmt.Println(left, "=", right); return nil}
}

func parseNum(l *lexer) fn {
	enter("num")
	defer exit("num")
	val, err := strconv.ParseFloat(l.str, 64)
	if err != nil {
		panic(err)
	}
	return func() interface{} { return val }
}

func parseArgs(l *lexer) (args []fn, err fn) {
	enter("args")
	defer exit("args")
	l.advance()
	for {
		switch l.typ {
		case RPAREN:
			return args, nil
		case NUM:
			args = append(args, parseNum(l))
		case IDENT:
			args = append(args, parseIdent(l))
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

func assert(test bool){
	if !test{panic("assertion failed")}
}
