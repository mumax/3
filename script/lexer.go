package script

import (
	"fmt"
	"io"
	"text/scanner"
)

type lexer struct {
	scanner.Scanner
	err error
}

func lex(src io.Reader) ([]*token, error) {
	l := newLexer(src)
	var tokens []*token
	for token := l.next(); token.typ != EOF; token = l.next() {
		if l.err != nil {
			return nil, l.err
		}
		if token.typ == ERR {
			return nil, fmt.Errorf("%v: illegal token: %v", token.Position, token.val)
		}
		if token.typ == EOL {
			token.val = ";"
		}
		tokens = append(tokens, token)
	}
	tokens = append(tokens, &token{EOL, ";", l.Position}) // add final endline
	return tokens, nil
}

func newLexer(src io.Reader) *lexer {
	l := new(lexer)
	l.Init(src)
	l.Whitespace = 1<<'\t' | 1<<' '
	l.Scanner.Error = func(s *scanner.Scanner, msg string) {
		l.err = fmt.Errorf("%v: syntax error: %v", l.Scanner.Position, msg)
	}
	return l
}

func (l *lexer) next() *token {
	tok := l.Scan()
	if tok == scanner.EOF {
		return &token{EOF, "", l.Position}
	} else {
		v := l.TokenText()
		return &token{typeof(v), v, l.Position}
	}
}
