package engine

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"io"
	"os"
)

var Table = *newTable("table") // output handle for tabular data (average magnetization etc.)

func init() {
	DeclFunc("TableAdd", TableAdd, "Add quantity as a column to the data table.")
	DeclFunc("TableAddVar", TableAddVariable, "Add user-defined variable + name + unit to data table.")
	DeclFunc("TableSave", TableSave, "Save the data table right now (appends one line).")
	DeclFunc("TableAutoSave", TableAutoSave, "Auto-save the data table every period (s). Zero disables save.")
	DeclFunc("TablePrint", TablePrint, "Print anyting in the data table")
	Table.Add(&M)
}

type DataTable struct {
	*bufio.Writer
	file io.Closer
	info
	outputs []TableData
	autosave
	history [][][]float64 // history for plot, indexed by: quantity, component, row
}

func newTable(name string) *DataTable {
	t := new(DataTable)
	t.name = name
	return t
}

func TableAdd(col TableData) {
	Table.Add(col)
}

func TableAddVariable(x script.ScalarFunction, name, unit string) {
	Table.AddVariable(x, name, unit)
}

func (t *DataTable) AddVariable(x script.ScalarFunction, name, unit string) {
	TableAdd(&userVar{x, name, unit})
}

type userVar struct {
	value      script.ScalarFunction
	name, unit string
}

func (x *userVar) Name() string       { return x.name }
func (x *userVar) NComp() int         { return 1 }
func (x *userVar) Unit() string       { return x.unit }
func (x *userVar) average() []float64 { return []float64{x.value.Float()} }

func TableSave() {
	Table.Save()
}

func TableAutoSave(period float64) {
	Table.autosave = autosave{period, Time, -1, nil} // count -1 allows output on t=0
}

func (t *DataTable) Add(output TableData) {
	if t.inited() {
		util.Fatal("data table add ", output.Name(), ": need to add quantity before table is output the first time")
	}
	t.outputs = append(t.outputs, output)
}

func (t *DataTable) Save() {
	t.init()
	_, err := fmt.Fprint(t, Time)
	util.FatalErr(err)
	t.history[0][0] = append(t.history[0][0], Time) // first column: time (1 component)
	for i, o := range t.outputs {
		vec := o.average()
		for c, v := range vec {
			_, err := fmt.Fprint(t, "\t", float32(v))
			util.FatalErr(err)
			t.history[i+1][c] = append(t.history[i+1][c], v)
		}
	}
	_, err = fmt.Fprintln(t)
	util.FatalErr(err)
	err = t.Flush()
	util.FatalErr(err)
	t.count++
}

func (t *DataTable) Println(msg ...interface{}) {
	t.init()
	_, err := fmt.Fprintln(t, msg...)
	util.FatalErr(err)
}

func TablePrint(msg ...interface{}) {
	Table.Println(msg...)
}

// open writer and write header
func (t *DataTable) init() {
	if !t.inited() {
		f, err := os.OpenFile(OD+t.name+".txt", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		util.FatalErr(err)
		t.Writer = bufio.NewWriter(f)
		t.file = f // so we can close it

		// write header
		_, err = fmt.Fprint(t, "# t (s)")
		util.FatalErr(err)
		for _, o := range t.outputs {
			if o.NComp() == 1 {
				_, err := fmt.Fprint(t, "\t", o.Name(), " (", o.Unit(), ")")
				util.FatalErr(err)
			} else {
				for c := 0; c < o.NComp(); c++ {
					_, err := fmt.Fprint(t, "\t", o.Name()+string('x'+c), " (", o.Unit(), ")")
					util.FatalErr(err)
				}
			}
		}
		_, err = fmt.Fprintln(t)
		util.FatalErr(err)
		err = t.Flush()
		util.FatalErr(err)

		// history for plot
		t.history = make([][][]float64, len(t.outputs)+1) // outputs + time column
		t.history[0] = make([][]float64, 1)               // time column (1 component)
		for q := range t.outputs {                        // other columns
			t.history[q+1] = make([][]float64, t.outputs[q].NComp())
		}
	}
}

func (t *DataTable) inited() bool {
	return t.Writer != nil
}

func (t *DataTable) flush() {
	if t.Writer != nil {
		err := t.Flush()
		util.FatalErr(err)
	}
}

func (t *DataTable) close() {
	t.flush()
	if t.file != nil {
		err := t.file.Close()
		util.FatalErr(err)
	}
}

// can be saved in table
type TableData interface {
	average() []float64
	Name() string
	Unit() string
	NComp() int
}
