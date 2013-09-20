package engine

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/util"
	"log"
	"os"
)

var Table = *newTable("datatable") // output handle for tabular data (average magnetization etc.)

func init() {
	DeclFunc("TableAdd", TableAdd, "Add quantity as a column to the data table.")
	DeclFunc("TableSave", TableSave, "Save the data table right now (appends one line).")
	DeclFunc("TableAutoSave", TableAutoSave, "Auto-save the data table ever period (s).")
}

type DataTable struct {
	*bufio.Writer
	doc
	outputs []TableData
	autosave
}

func newTable(name string) *DataTable {
	t := new(DataTable)
	t.name = name
	return t
}

func TableAdd(col TableData) {
	Table.Add(col)
}

func TableSave() {
	Table.Save()
}

func TableAutoSave(period float64) {
	Table.autosave = autosave{period, Time, 0}
}

func (t *DataTable) Add(output TableData) {
	if t.inited() {
		log.Fatalln("data table add", output.Name(), ": need to add quantity before table is output the first time")
	}
	t.outputs = append(t.outputs, output)
}

//func (t *DataTable) AddFunc(nComp int, name, unit string, f func() []float64) {
//	t.Add(newGetfunc(nComp, name, unit, "", f))
//}

func (t *DataTable) Save() {
	t.init()
	fmt.Fprint(t, Time)
	for _, o := range t.outputs {
		vec := o.TableData()
		for _, v := range vec {
			fmt.Fprint(t, "\t", v)
		}
	}
	fmt.Fprintln(t)
	t.Flush()
	t.count++
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

// can be saved in table
type TableData interface {
	TableData() []float64 // TODO: output float32s only
	Name() string
	Unit() string
	NComp() int
}
