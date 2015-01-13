package engine

import (
	"fmt"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/script"
	"github.com/mumax/3/util"
	"io"
	"sync"
	"time"
)

var Table = *newTable("table") // output handle for tabular data (average magnetization etc.)

func init() {
	DeclFunc("TableAdd", TableAdd, "Add quantity as a column to the data table.")
	DeclFunc("TableAddVar", TableAddVariable, "Add user-defined variable + name + unit to data table.")
	DeclFunc("TableSave", TableSave, "Save the data table right now (appends one line).")
	DeclFunc("TableAutoSave", TableAutoSave, "Auto-save the data table every period (s). Zero disables save.")
	DeclFunc("TableAutoSaveStep", TableAutoSaveStep, "Auto-save the data table every steps. Zero disables save.")
	DeclFunc("TablePrint", TablePrint, "Print anyting in the data table")
	Table.Add(&M)
}

type DataTable struct {
	output interface {
		io.Writer
		Flush() error
	}
	info
	outputs []TableData
	autosave
	flushlock sync.Mutex
}

func (t *DataTable) Write(p []byte) (int, error) {
	n, err := t.output.Write(p)
	util.FatalErr(err)
	return n, err
}

func (t *DataTable) Flush() error {
	if t.output == nil {
		return nil
	}
	err := t.output.Flush()
	util.FatalErr(err)
	return err
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
	nbr_steps := 0
	Table.autosave = autosave{period, nbr_steps, Time, -1, nil} // count -1 allows output on t=0
}

func TableAutoSaveStep(nbr_steps int) {
	period := 0.0
	Table.autosave = autosave{period, nbr_steps, Time, -1, nil} // count -1 allows output on t=0
}

func (t *DataTable) Add(output TableData) {
	if t.inited() {
		util.Fatal("data table add ", output.Name(), ": need to add quantity before table is output the first time")
	}
	t.outputs = append(t.outputs, output)
}

func (t *DataTable) Save() {
	t.init()
	fmt.Fprint(t, Time)
	for _, o := range t.outputs {
		vec := o.average()
		for _, v := range vec {
			fmt.Fprint(t, "\t", float32(v))
		}
	}
	fmt.Fprintln(t)
	t.flush()
	t.count++
}

func (t *DataTable) Println(msg ...interface{}) {
	t.init()
	fmt.Fprintln(t, msg...)
}

func TablePrint(msg ...interface{}) {
	Table.Println(msg...)
}

// open writer and write header
func (t *DataTable) init() {
	if t.inited() {
		return
	}
	f, err := httpfs.Create(OD() + t.name + ".txt")
	util.FatalErr(err)
	t.output = f

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

	go func() {
		time.Sleep(1 * time.Second)
		Table.flush()
	}()
}

func (t *DataTable) inited() bool {
	return t.output != nil
}

func (t *DataTable) flush() {
	t.flushlock.Lock()
	defer t.flushlock.Unlock()
	t.Flush()
}

// can be saved in table
type TableData interface {
	average() []float64
	Name() string
	Unit() string
	NComp() int
}
