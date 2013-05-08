package script

import (
	"fmt"
	"io"
	"text/scanner"
)

type lexer struct {
	scn scanner.Scanner
	str string    // last read token
	typ tokenType // last read token type
	scanner.Position
	peekStr string    // peek-ahead value for str
	peekTyp tokenType // peek-ahead value for typ
}

func newLexer() *lexer {
	l := new(lexer)
	return l
}

func (l *lexer) init(src io.Reader) {
	l.scn.Init(src)
	l.scn.Whitespace = 1<<'\t' | 1<<' '
	l.scn.Error = func(s *scanner.Scanner, msg string) {
		l.peekStr = fmt.Sprintf("%v: syntax error: %v", l.Position, msg)
		l.peekTyp = ERR
	}
	l.scn.Scan() // peek
	l.peekStr = l.scn.TokenText()
	l.peekTyp = typeof(l.peekStr)
}

func (l *lexer) unexpected() error {
	if l.typ == ERR {
		return fmt.Errorf("%v: syntax error: %v", l.Position, l.str)
	} else {
		return fmt.Errorf("%v: unexpected %v: %v", l.Position, l.typ, l.str)
	}
}

func (l *lexer) advance() {
	l.str = l.peekStr
	l.typ = l.peekTyp
	l.Position = l.scn.Pos() // peeked position
	l.scn.Scan()
	l.peekStr = l.scn.TokenText()
	l.peekTyp = typeof(l.peekStr)
}
