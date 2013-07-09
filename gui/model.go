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
	Setter
	Getter
}

type Caller interface {
	Call()
}

func (s *Server) Add(name string, model interface{}) {
	switch model.(type) {
	case Getter, Setter, Caller:
		s.addMod(name, model)
	}

	t := reflect.TypeOf(model)
	v := reflect.ValueOf(model)
	switch t.Kind() {
	case reflect.Func:
		if t.NumOut() == 1 && t.NumIn() == 0 { // getter
			s.addMod(name, &getter{func() interface{} { return v.Call([]reflect.Value{})[0].Interface() }})
			return
		}
		if t.NumOut() == 0 && t.NumIn() == 0 { // niladic call
			s.addMod(name, &caller{func() { v.Call([]reflect.Value{}) }})
			return
		}
	}
	panic(fmt.Sprint("server.add: can not handle model of type ", t))
}

type getter struct{ get func() interface{} }
type setter struct{ set func(interface{}) }
type caller struct{ f func() }

func (g *getter) Get() interface{}  { return g.get() }
func (s *setter) Set(v interface{}) { s.set(v) }
func (c *caller) Call()             { c.f() }

type model struct {
	setter
	getter
}

func (s *Server) addMod(name string, mod interface{}) {
	if _, ok := s.model[name]; ok {
		panic("model name " + name + " already in use")
	}
	s.model[name] = mod
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
