package script

import (
	"fmt"
	"io"
	"log"
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

// First compiles the entire source, then executes it.
// Compile errors are spotted before
func (p *Parser) Exec() (err error) {
	defer func() {
		panc := recover()
		if panc != nil {
			err = fmt.Errorf("%v", panc)
		}
	}()

	var code []Expr
	expr, err := p.parseLine()
	for err != io.EOF {
		if err != nil {
			return err
		}
		code = append(code, expr)
		expr, err = p.parseLine()
	}

	for _, e := range code {
		ret := e.Eval()
		log.Println("eval", e, ":", ret)
	}
	return nil
}

func (p *Parser) parseLine() (ex Expr, err error) {

	p.advance()
	switch p.typ {
	case EOF:
		return nil, io.EOF // marks end of input
	case EOL:
		return &nop{}, nil // empty line
	default:
		Expr := p.parseExpr()
		p.advance()
		if p.typ == EOL || p.typ == EOF { // statement has to be terminated
			return Expr, nil
		} else {
			panic(p.unexpected())
		}
	}
}

func (p *Parser) parseIdent() Expr {
	switch p.peekTyp {
	case LPAREN:
		return p.parseCall()
	case ASSIGN:
		return p.parseAssign()
	default:
		return p.getvar(p.str)
	}
}

func (p *Parser) parseExpr() Expr {
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

func (p *Parser) parseCall() Expr {
	funcname := p.str
	p.advance()
	assert(p.typ == LPAREN)
	args := p.parseArgs()
	return &call{funcname, args}
}

func (p *Parser) parseAssign() Expr {
	left := p.str
	p.advance()
	assert(p.typ == ASSIGN)
	p.advance()
	right := p.parseExpr()
	return p.newAssign(left, right)
}

func (p *Parser) parseNum() Expr {
	val, err := strconv.ParseFloat(p.str, 64)
	if err != nil {
		panic(err)
	}
	return num(val)
}

func (p *Parser) parseArgs() []Expr {
	var args []Expr
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
