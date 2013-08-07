package main

import (
	"code.google.com/p/mx3/cuda"
	"code.google.com/p/mx3/engine"
	"fmt"
	"reflect"
	"sort"
)

func main() {

	cuda.Init()
	cuda.LockThread()

	ident := engine.World.Identifiers
	doc := engine.World.Doc
	e := make(entries, 0, len(ident))
	for k, v := range ident {
		e = append(e, entry{k, v.Type(), doc[k]})
	}

	sort.Sort(&e)
	for _, x := range e {
		fmt.Println(x)
	}
}

type entry struct {
	name string
	typ  reflect.Type
	doc  string
}

type entries []entry

func (e *entries) Len() int           { return len(*e) }
func (e *entries) Less(i, j int) bool { return (*e)[i].name < (*e)[j].name }
func (e *entries) Swap(i, j int)      { (*e)[i], (*e)[j] = (*e)[j], (*e)[i] }
