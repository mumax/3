package script

import (
	"fmt"
	"io"
	"strconv"
)

type parser struct {
	*lexer
	world
}

func newParser(src io.Reader) parser {
	p := parser{lexer: newLexer(src)}
	p.world.init()
	return p
}

func (p *parser) Exec() error {
	expr, err := p.ParseLine()
	for err != io.EOF {
		if err == nil {
			fmt.Println("eval", expr, ":", expr.eval())
		} else {
			fmt.Println("err:", err)
			return err
		}
		expr, err = p.ParseLine()
	}
	return nil
}

func (p *parser) ParseLine() (ex expr, err error) {
	defer func() {
		panc := recover()
		if panc != nil {
			ex = nil
			err = fmt.Errorf("%v", panc)
			// skip rest of line
			for p.typ != EOF && p.typ != EOL {
				p.advance()
			}
		}
	}()

	p.advance()
	switch p.typ {
	case EOF:
		return nil, io.EOF // marks end of input
	case EOL:
		return &nop{}, nil // empty line
	default:
		expr := p.parseExpr()
		p.advance()
		if p.typ == EOL || p.typ == EOF { // statement has to be terminated
			return expr, nil
		} else {
			panic(p.unexpected())
		}
	}
}

func (p *parser) parseIdent() expr {
	switch p.peekTyp {
	case LPAREN:
		return p.parseCall()
	case ASSIGN:
		return p.parseAssign()
	default:
		return p.getvar(p.str)
	}
}

func (p *parser) parseExpr() expr {
	switch p.typ {
	case IDENT:
		return p.parseIdent()
	case NUM:
		return p.parseNum()
	default:
		panic(p.unexpected())
		// TODO: handle parens, commas
	}
}

func (p *parser) parseCall() expr {
	funcname := p.str
	p.advance()
	assert(p.typ == LPAREN)
	args := p.parseArgs()
	return &call{funcname, args}
}

func (p *parser) parseAssign() expr {
	left := p.str
	p.advance()
	assert(p.typ == ASSIGN)
	p.advance()
	right := p.parseExpr()
	return p.newAssign(left, right)
}

func (p *parser) parseNum() expr {
	val, err := strconv.ParseFloat(p.str, 64)
	if err != nil {
		panic(err)
	}
	return num(val)
}

func (p *parser) parseArgs() []expr {
	var args []expr
	p.advance()
	for {
		switch p.typ {
		case RPAREN:
			return args
		case NUM:
			args = append(args, p.parseNum())
		case IDENT:
			args = append(args, p.parseIdent())
		default:
			panic(p.unexpected())
		}
		p.advance()
		if p.typ == COMMA {
			p.advance()
			if p.typ == RPAREN {
				panic(p.unexpected())
			}
		}
	}
}

func assert(test bool) {
	if !test {
		panic("assertion failed")
	}
}
