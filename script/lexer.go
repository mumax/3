package script

import (
	"fmt"
	"io"
	"text/scanner"
)

type lexer struct {
	scan    scanner.Scanner
	str     string    // last read token
	typ     tokenType // last read token type
	peekStr string    // peek-ahead value for str
	peekTyp tokenType // peek-ahead value for typ
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.scan.Init(src)
	l.scan.Whitespace = 1<<'\t' | 1<<' '
	l.scan.Error = func(s *scanner.Scanner, msg string) {
		l.peekStr = fmt.Sprintf("%v: syntax error: %v", l.scan.Position, msg)
		l.peekTyp = ERR
	}
	l.scan.Scan() // peek
	l.peekStr = l.scan.TokenText()
	l.peekTyp = typeof(l.peekStr)
	return l
}

func (l *lexer) unexpected() fn {
	err := ""
	if l.typ == ERR {
		err = fmt.Sprint(l.scan.Pos(), ":syntax error:", l.str)
	} else {
		err = fmt.Sprint(l.scan.Pos(), ":unexpected:", l.typ, ":", l.str)
	}
	return func() interface{} { return err }
}

func (l *lexer) advance() {
	l.str = l.peekStr
	l.typ = l.peekTyp
	fmt.Println("--advance:", l.typ, ":", l.str)
	l.scan.Scan()
	l.peekStr = l.scan.TokenText()
	l.peekTyp = typeof(l.peekStr)
}
