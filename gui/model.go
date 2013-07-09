package gui

import (
	"fmt"
	"reflect"
)

type Setter interface {
	Set(interface{})
}

type Getter interface {
	Get() interface{}
}

type Model interface {
	Getter
	Setter
}

func (s *Server) Add(name string, model interface{}) {
	t := reflect.TypeOf(model)
	switch m := t.(type) {
	case Model:
		s.addMod(name, m.Get, m.Set)
		return
	case Setter:
		s.addMod(name, nil, m.Set)
		return
	case Getter:
		s.addMod(name, m.Get, nil)
		return
	}

	v := reflect.ValueOf(model)
	switch t.Kind() {
	case reflect.Func:
		if t.NumOut() == 1 && t.NumIn() == 0 {
			s.addMod(name, func() interface{} { return v.Call([]reflect.Value{})[0].Interface() }, nil)
			return
		}
	}
	panic(fmt.Sprint("server.add: can not handle model of type ", t))
}

type mod struct {
	name string
	get  func() interface{}
	set  func(interface{})
}

func (m *mod) Get() interface{}  { return m.get() }
func (m *mod) Set(v interface{}) { m.set(v) }

func (s *Server) addMod(name string, get func() interface{}, set func(interface{})) {
	if _, ok := s.model[name]; ok {
		panic("model name " + name + " already in use")
	}
	s.model[name] = &mod{name, get, set}
}

//
//
//func (m *method_) String() string {
//	args := []reflect.Value{}
//	r := m.Call(args)
//	argument(len(r) == 1, "need one return value")
//	return fmt.Sprint(r[0].Interface())
//}
//
//type method_ struct{ reflect.Value }
//
//func method(v reflect.Value, field string) fmt.Stringer {
//	m := v.MethodByName(field)
//	if m.Kind() != 0 {
//		return &method_{m}
//	} else {
//		panic(fmt.Sprint("type ", v.Type(), " has no such field or method ", field))
//	}
//}
//
//func (v *Server) addElem(o fmt.Stringer) (id int) {
//	v.elements = append(v.elements, o)
//	return len(v.elements)
//}
