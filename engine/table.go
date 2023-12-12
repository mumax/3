package engine

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/script"
	"github.com/mumax/3/timer"
	"github.com/mumax/3/util"
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

type ColumnHeader struct {
	name string
	unit string
}

func (c ColumnHeader) String() string {
	return c.name + " (" + c.unit + ")"
}

func (c ColumnHeader) Name() string {
	return c.name
}

func (c ColumnHeader) Unit() string {
	return c.unit
}

func (t *DataTable) Header() (headers []ColumnHeader) {
	headers = make([]ColumnHeader, 0)
	headers = append(headers, ColumnHeader{"t", "s"})
	for _, o := range t.outputs {
		if o.NComp() == 1 {
			headers = append(headers, ColumnHeader{NameOf(o), UnitOf(o)})
		} else {
			for c := 0; c < o.NComp(); c++ {
				name := NameOf(o) + string('x'+c)
				headers = append(headers, ColumnHeader{name, UnitOf(o)})
			}
		}
	}
	return
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

func (t *DataTable) Read() (data [][]float64, err error) {
	if !t.inited() {
		return nil, errors.New("Table is not initialized")
	}

	t.flush()
	t.flushlock.Lock()
	defer t.flushlock.Unlock()

	rawdata, err := httpfs.Read(fmt.Sprintf(`%v%s.txt`, OD(), t.name))
	if err != nil {
		return
	}

	csvReader := csv.NewReader(bytes.NewReader(rawdata))
	csvReader.Comma = '\t'
	csvReader.Comment = '#'

	nCols := len(t.Header())
	data = make([][]float64, 0)

	for {
		line, err := csvReader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		record := make([]float64, nCols)
		for c, _ := range record {
			record[c], err = strconv.ParseFloat(strings.TrimSpace(line[c]), 64)
			if err != nil {
				return nil, err
			}
		}

		data = append(data, record)
	}

	return data, nil
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
	header := t.Header()
	fprint(t, "# ", header[0])
	for col := 1; col < len(header); col++ {
		fprint(t, "\t", header[col])
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
