package engine

import (
	"fmt"
	"github.com/mumax/3/v3/cuda"
	"github.com/mumax/3/v3/data"
	"github.com/mumax/3/v3/httpfs"
	"github.com/mumax/3/v3/script"
	"github.com/mumax/3/v3/timer"
	"github.com/mumax/3/v3/util"
	"io"
	"sync"
	"time"
)

var Table = *newTable("table") // output handle for tabular data (average magnetization etc.)
const TableAutoflushRate = 5   // auto-flush table every X seconds

func init() {
	DeclFunc("TableAdd", TableAdd, "Add quantity as a column to the data table.")
	DeclFunc("TableAddVar", TableAddVariable, "Add user-defined variable + name + unit to data table.")
	DeclFunc("TableSave", TableSave, "Save the data table right now (appends one line).")
	DeclFunc("TableAutoSave", TableAutoSave, "Auto-save the data table every period (s). Zero disables save.")
	DeclFunc("TablePrint", TablePrint, "Print anyting in the data table")
	Table.Add(&M)
}

type DataTable struct {
	output interface {
		io.Writer
		Flush() error
	}
	info
	outputs []Quantity
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

	if cuda.Synchronous {
		timer.Start("io")
	}
	err := t.output.Flush()
	if cuda.Synchronous {
		timer.Stop("io")
	}
	util.FatalErr(err)
	return err
}

func newTable(name string) *DataTable {
	t := new(DataTable)
	t.name = name
	return t
}

func TableAdd(col Quantity) {
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
func (x *userVar) EvalTo(dst *data.Slice) {
	avg := x.average()
	for c := 0; c < x.NComp(); c++ {
		cuda.Memset(dst.Comp(c), float32(avg[c]))
	}
}

func TableSave() {
	Table.Save()
}

func TableAutoSave(period float64) {
	Table.autosave = autosave{period, Time, -1, nil} // count -1 allows output on t=0
}

func (t *DataTable) Add(output Quantity) {
	if t.inited() {
		util.Fatal("data table add ", NameOf(output), ": need to add quantity before table is output the first time")
	}
	t.outputs = append(t.outputs, output)
}

func (t *DataTable) Save() {
	t.flushlock.Lock() // flush during write gives errShortWrite
	defer t.flushlock.Unlock()

	if cuda.Synchronous {
		timer.Start("io")
	}
	t.init()
	fprint(t, Time)
	for _, o := range t.outputs {
		vec := AverageOf(o)
		for _, v := range vec {
			fprint(t, "\t", float32(v))
		}
	}
	fprintln(t)
	//t.flush()
	t.count++

	if cuda.Synchronous {
		timer.Stop("io")
	}
}

func (t *DataTable) Println(msg ...interface{}) {
	t.init()
	fprintln(t, msg...)
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
	fprint(t, "# t (s)")
	for _, o := range t.outputs {
		if o.NComp() == 1 {
			fprint(t, "\t", NameOf(o), " (", UnitOf(o), ")")
		} else {
			for c := 0; c < o.NComp(); c++ {
				fprint(t, "\t", NameOf(o)+string('x'+c), " (", UnitOf(o), ")")
			}
		}
	}
	fprintln(t)
	t.Flush()

	// periodically flush so GUI shows graph,
	// but don't flush after every output for performance
	// (httpfs flush is expensive)
	go func() {
		for {
			time.Sleep(TableAutoflushRate * time.Second)
			Table.flush()
		}
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

// Safe fmt.Fprint, will fail on error
func fprint(out io.Writer, x ...interface{}) {
	_, err := fmt.Fprint(out, x...)
	util.FatalErr(err)
}

// Safe fmt.Fprintln, will fail on error
func fprintln(out io.Writer, x ...interface{}) {
	_, err := fmt.Fprintln(out, x...)
	util.FatalErr(err)

}
