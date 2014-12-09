package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff vSetter // total effective field

func init() { B_eff.init("B_eff", "T", "Effective field", SetEffectiveField) }

// Sets dst to the current effective field (T).
func SetEffectiveField(dst *data.Slice) {
	B_demag.Set(dst)  // set to B_demag...
	B_exch.AddTo(dst) // ...then add other terms
	B_anis.AddTo(dst)
	B_ext.AddTo(dst)
	if !relaxing {
		B_therm.AddTo(dst)
	}
}
