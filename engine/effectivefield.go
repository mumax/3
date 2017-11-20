package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff = NewVectorField("B_eff", "T", "Effective field", SetEffectiveField)

// Sets dst to the current effective field, in Tesla.
// This is the sum of all effective field terms,
// like demag, exchange, ...
func SetEffectiveField(dst *data.Slice) {
	SetDemagField(dst)    // set to B_demag...
	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)
	B_ext.AddTo(dst)
	if !relaxing {
		if ((solvertype!=7)&&(solvertype!=8)){
                	B_therm.AddTo(dst)
                }
	}
	AddCustomField(dst)
}
