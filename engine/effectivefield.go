package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff vSetter // total effective field

func init() { B_eff.init("B_eff", "T", "Effective field", SetEffectiveField) }

// Sets dst to the current effective field (T).
func SetEffectiveField(dst *data.Slice) {
	SetDemagField(dst)    // set to B_demag...
	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)
	B_ext.AddTo(dst)
	if !relaxing {
		B_therm.AddTo(dst)
	}
}
