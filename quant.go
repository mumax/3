package main

import ()

type Quant interface {
	Name() string
	Unit() string
	Get(comp, index int) float32
}

// Abstract Quantity, has only name+unit.
type AQuant struct {
	name string
	unit string
}

func (this *AQuant) Init() { }

func (this *AQuant) Name() string { return this.name }

func (this *AQuant) Unit() string { return this.unit }
