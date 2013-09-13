package engine

import (
	"bufio"
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"os"
)

type DataTable struct {
	*bufio.Writer
	doc
	outputs []getVec
}

// can be saved in table
type getVec interface {
	GetVec() []float64
	Name() string
	Unit() string
	NComp() int
}

func (t *DataTable) Add(output getVec) {
	if t.inited() {
		log.Fatalln("data table add", output.Name(), ": need to add quantity before table is output the first time")
	}
	t.outputs = append(t.outputs, output)
}

//func (t *DataTable) AddFunc(nComp int, name, unit string, f func() []float64) {
//	t.Add(newScalar(nComp, name, unit, f))
//}

func (t *DataTable) Save() {
	t.init()
	fmt.Fprint(t, Time)
	for _, o := range t.outputs {
		vec := o.GetVec()
		for _, v := range vec {
			fmt.Fprint(t, "\t", v)
		}
	}
	fmt.Fprintln(t)
	t.Flush()
}

func (t *DataTable) AutoSave(period float64) {
	AutoSave(t, period)
}

func newTable(name string) *DataTable {
	t := new(DataTable)
	t.name = name
	return t
}

// open writer and write header
func (t *DataTable) init() {
	if !t.inited() {
		f, err := os.OpenFile(OD+t.name+".txt", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		util.FatalErr(err)
		t.Writer = bufio.NewWriter(f)

		// write header
		fmt.Fprint(t, "# t (s)")
		for _, o := range t.outputs {
			if o.NComp() == 1 {
				fmt.Fprint(t, "\t", o.Name(), " (", o.Unit(), ")")
			} else {
				for c := 0; c < o.NComp(); c++ {
					fmt.Fprint(t, "\t", o.Name()+string('x'+c), " (", o.Unit(), ")")
				}
			}
		}
		fmt.Fprintln(t)
		t.Flush()
	}
}

func (t *DataTable) inited() bool {
	return t.Writer != nil
}

func (t *DataTable) flush() {
	if t.Writer != nil {
		t.Flush()
	}
}
