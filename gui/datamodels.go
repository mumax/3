package gui

import (
	"fmt"
	"log"
	"strconv"
)

type interfaceData struct {
	v interface{}
}

func (d *interfaceData) setValue(v interface{}) {
	d.v = v
}

func (d *interfaceData) value() interface{} {
	return d.v
}

type boolData struct{ v bool }
type intData struct{ v int }
type floatData struct{ v float64 }

func (d *boolData) value() interface{}  { return d.v }
func (d *intData) value() interface{}   { return d.v }
func (d *floatData) value() interface{} { return d.v }

func (d *boolData) setValue(v interface{}) {
	d.v = v.(bool)
}

func (d *intData) setValue(v interface{}) {
	switch v := v.(type) {
	case int:
		d.v = v
	default:
		i, err := strconv.Atoi(fmt.Sprint(v))
		if err == nil {
			d.v = i
		} else {
			log.Println(err)
		}
	}
}

func (d *floatData) setValue(v interface{}) {
	switch v := v.(type) {
	case float64:
		d.v = v
	default:
		i, err := strconv.ParseFloat(fmt.Sprint(v), 64)
		if err == nil {
			d.v = i
		} else {
			log.Println(err)
		}
	}
}
