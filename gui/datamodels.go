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

type boolData struct{ interfaceData }

func (d *boolData) setValue(v interface{}) {
	d.v = v.(bool)
}

func BoolData(v bool) *boolData {
	return &boolData{interfaceData{v}}
}

type intData struct{ interfaceData }

func IntData(v int) *intData {
	return &intData{interfaceData{v}}
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

type floatData struct{ interfaceData }

func FloatData(v float64) *floatData {
	return &floatData{interfaceData{v}}
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
