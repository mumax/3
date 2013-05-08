package script

import (
	"fmt"
	"io"
	"strconv"
)

type Parser struct {
	*lexer
	world
}

func NewParser(src io.Reader) Parser {
	p := Parser{lexer: newLexer(src)}
	p.world.init()
	return p
}

func (p *Parser) Exec() error {
	expr, err := p.parseLine()
	for err != io.EOF {
		if err == nil {
			fmt.Println("eval", expr, ":", expr.eval())
		} else {
			fmt.Println("err:", err)
			return err
		}
		expr, err = p.parseLine()
	}
	return nil
}

func (p *Parser) parseLine() (ex expr, err error) {
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

func (p *Parser) parseIdent() expr {
	switch p.peekTyp {
	case LPAREN:
		return p.parseCall()
	case ASSIGN:
		return p.parseAssign()
	default:
		return p.getvar(p.str)
	}
}

func (p *Parser) parseExpr() expr {
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

func (p *Parser) parseCall() expr {
	funcname := p.str
	p.advance()
	assert(p.typ == LPAREN)
	args := p.parseArgs()
	return &call{funcname, args}
}

func (p *Parser) parseAssign() expr {
	left := p.str
	p.advance()
	assert(p.typ == ASSIGN)
	p.advance()
	right := p.parseExpr()
	return p.newAssign(left, right)
}

func (p *Parser) parseNum() expr {
	val, err := strconv.ParseFloat(p.str, 64)
	if err != nil {
		panic(err)
	}
	return num(val)
}

func (p *Parser) parseArgs() []expr {
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
