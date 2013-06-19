package engine

//import (
//	"bufio"
//	"code.google.com/p/mx3/util"
//	"fmt"
//	"log"
//	"os"
//)
//
//type DataTable struct {
//	*bufio.Writer
//	autosave
//	outputs []*scalar
//}
//
//func (t *DataTable) Add(output *scalar) {
//	if t.inited() {
//		log.Fatalln("data table add", output.name, ": need to add quantity before table is output the first time")
//	}
//	t.outputs = append(t.outputs, output)
//}
//
//func (t *DataTable) AddFunc(nComp int, name, unit string, f func() []float64) {
//	t.Add(newScalar(nComp, name, unit, f))
//}
//
//func (t *DataTable) arm(good bool) {
//	if good && t.needSave() {
//		t.init()
//		for _, o := range t.outputs {
//			o.arm()
//		}
//	}
//}
//
//func (t *DataTable) touch(good bool) {
//	if good && t.needSave() {
//		fmt.Fprint(t, Time)
//		for _, o := range t.outputs {
//			vec := o.Get()
//			for _, v := range vec {
//				fmt.Fprint(t, "\t", v)
//			}
//		}
//		fmt.Fprintln(t)
//		t.Flush()
//		t.saved()
//	}
//}
//
//func newTable(name string) *DataTable {
//	t := new(DataTable)
//	t.name = name
//	return t
//}
//
//// open writer and write header
//func (t *DataTable) init() {
//	if !t.inited() {
//		f, err := os.OpenFile(OD+t.name+".txt", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
//		util.FatalErr(err)
//		t.Writer = bufio.NewWriter(f)
//
//		// write header
//		fmt.Fprint(t, "# t(s)")
//		for _, o := range t.outputs {
//			if o.nComp == 1 {
//				fmt.Fprint(t, "\t", o.name, " (", o.unit, ")")
//			} else {
//				for c := 0; c < o.nComp; c++ {
//					fmt.Fprint(t, "\t", o.name+string('x'+c), " (", o.unit, ")")
//				}
//			}
//		}
//		fmt.Fprintln(t)
//		t.Flush()
//	}
//}
//
//func (t *DataTable) inited() bool {
//	return t.Writer != nil
//}
//
//func (t *DataTable) flush() {
//	if t.Writer != nil {
//		t.Flush()
//	}
//}
