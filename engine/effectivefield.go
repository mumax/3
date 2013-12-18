package engine

import "github.com/mumax/3/data"

var B_eff setter // total effective field

func init() { B_eff.init(VECTOR, "B_eff", "T", "Effective field", SetEffectiveField) }

// Sets dst to the current effective field (T).
func SetEffectiveField(dst *data.Slice) {
	B_demag.Set(dst)  // set to B_demag...
	B_exch.AddTo(dst) // ...then add other terms
	B_anis.AddTo(dst)
	B_ext.AddTo(dst)
	B_therm.AddTo(dst)
}
