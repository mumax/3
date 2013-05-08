package script

import (
	"fmt"
	"io"
	"strconv"
)

func parseLine(l *lexer) (ex expr, err error) {
	defer func() {
		panc := recover()
		if panc != nil {
			ex = nil
			err = fmt.Errorf("%v", panc)
			// skip rest of line
			for l.typ != EOF && l.typ != EOL {
				l.advance()
			}
		}
	}()

	l.advance()
	switch l.typ {
	case EOF:
		return nil, io.EOF // marks end of input
	case EOL:
		return &nop{}, nil // empty line
	default:
		expr := parseExpr(l)
		l.advance()
		if l.typ == EOL || l.typ == EOF { // statement has to be terminated
			return expr, nil
		} else {
			panic(l.unexpected())
		}
	}
}

func parseIdent(l *lexer) expr {
	switch l.peekTyp {
	case LPAREN:
		return parseCall(l)
	case ASSIGN:
		return parseAssign(l)
	default:
		return &variable{l.str}
	}
}

func parseExpr(l *lexer) expr {
	switch l.typ {
	case IDENT:
		return parseIdent(l)
	case NUM:
		return parseNum(l)
	default:
		panic(l.unexpected())
		// TODO: handle parens, commas
	}
}

func parseCall(l *lexer) expr {
	funcname := l.str
	l.advance()
	assert(l.typ == LPAREN)
	args := parseArgs(l)
	return &call{funcname, args}
}

func parseAssign(l *lexer) expr {
	left := l.str
	l.advance()
	assert(l.typ == ASSIGN)
	l.advance()
	right := parseExpr(l)
	return &assign{left, right}
}

func parseNum(l *lexer) expr {
	val, err := strconv.ParseFloat(l.str, 64)
	if err != nil {
		panic(err)
	}
	return num(val)
}

func parseArgs(l *lexer) []expr {
	var args []expr
	l.advance()
	for {
		switch l.typ {
		case RPAREN:
			return args
		case NUM:
			args = append(args, parseNum(l))
		case IDENT:
			args = append(args, parseIdent(l))
		default:
			panic(l.unexpected())
		}
		l.advance()
		if l.typ == COMMA {
			l.advance()
			if l.typ == RPAREN {
				panic(l.unexpected())
			}
		}
	}
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}
