package engine

// Handlers for parameter setting through web interface

import (
	"code.google.com/p/mx3/util"
	"fmt"
	"log"
	"net/http"
	"strconv"
)

// handle to set numerical parameters
func setparam(w http.ResponseWriter, r *http.Request) {

	injectAndWait(func() {
		for name, inf := range params {
			vals := make([]float64, inf.NComp())
			for c := range inf.Comp() {
				str := r.FormValue(fmt.Sprint(name, c))
				v, err := strconv.ParseFloat(str, 64)
				if err != nil {
					http.Error(w, "set "+name+": "+err.Error(), 400)
					return
				}
				vals[c] = v
			}
			have := inf.Get()
			if !eq(have, vals) {
				log.Println("set", name, vals)
				inf.Set(vals)
			}
		}
	})

	http.Redirect(w, r, "/", http.StatusFound)
}

func eq(a, b []float64) bool {
	for i, v := range a {
		if b[i] != v {
			return false
		}
	}
	return true
}

var params = map[string]param{
	"aex":     {"J/m", "Exchange stiffness", &Aex},
	"msat":    {"A/m", "Saturation magnetization", &Msat},
	"alpha":   {"", "Damping constant", &Alpha},
	"b_ext":   {"T", "External field", &B_ext},
	"dmi":     {"J/m2", "Dzyaloshinskii-Moriya strength", &DMI},
	"ku1":     {"J/m3", "Uniaxial anisotropy vector", &Ku1},
	"xi":      {"", "Spin-transfer torque", &Xi}, // TODO: replace by beta
	"spinpol": {"", "Spin polarization", &SpinPol},
	"j":       {"A/m2", "Electrical current density", &J},
}

type param struct {
	Unit, Descr string
	Handle      interface{}
}

func (m param) Comp() []int {
	switch m.Handle.(type) {
	default:
		panic("meta-inf: unknown pointer type")
	case *ScalFn:
		return []int{0}
	case *VecFn:
		return []int{0, 1, 2}
	}
}

func (m param) NComp() int { return len(m.Comp()) }

func (m param) GetComp(comp int) float64 {
	return m.Get()[comp]
}

func (m param) Get() []float64 {
	switch h := m.Handle.(type) {
	default:
		panic("meta-inf: unknown pointer type")
	case *ScalFn:
		if *h == nil {
			return []float64{0}
		}
		return []float64{(*h)()}
	case *VecFn:
		if *h == nil {
			return []float64{0, 0, 0}
		}
		v := (*h)()
		return v[:]
	}
}

func (m param) Set(v []float64) {
	switch h := m.Handle.(type) {
	default:
		panic("meta-inf: unknown pointer type")
	case *ScalFn:
		util.Argument(len(v) == 1)
		*h = Const(v[0])
	case *VecFn:
		util.Argument(len(v) == 3)
		*h = ConstVector(v[0], v[1], v[2])
	}
}

func (ui *guistate) Params() map[string]param { return params }
