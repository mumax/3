package engine

//import (
//	//"code.google.com/p/mx3/cuda"
//	//"code.google.com/p/mx3/data"
//	"fmt"
//)
//
////type adder struct {
////	addTo func(dst *data.Slice) // adds quantity to dst
////	need bool
////}
////
////type Handle struct{
////	autosave
////}
////
////func newAdder(adder func(dst *data.Slice)) adder {
////	return adder{adder, false}
////}
////
////func (q *adder) AddTo(dst *data.Slice) {
////	if q.need {
////		buffer := outputBuffer(dst.NComp())
////		q.addTo(buffer)
////		cuda.Madd2(dst, dst, buffer, 1, 1)
////		go saveAndRecycle(buffer, q.fname(), Time)
////		q.autosave.count++ // !
////	} else {
////		q.addTo(dst)
////	}
////}
//
