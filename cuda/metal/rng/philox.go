// Philox-derived portions: Copyright 2010-2012, D. E. Shaw Research.
// BSD-3-Clause; see the Random123 notice in the repository LICENSE.

// Package rng provides the Metal thermal-noise generator and a pure-Go
// Random123-compatible Philox implementation used for known-answer tests.
package rng

import "math"

const (
	philoxM0 = uint32(0xD2511F53)
	philoxM1 = uint32(0xCD9E8D57)
	philoxW0 = uint32(0x9E3779B9)
	philoxW1 = uint32(0xBB67AE85)
)

// Philox4x32 applies ten Random123 Philox rounds.
func Philox4x32(counter [4]uint32, key [2]uint32) [4]uint32 {
	for round := 0; round < 10; round++ {
		counter = philoxRound(counter, key)
		key[0] += philoxW0
		key[1] += philoxW1
	}
	return counter
}

func philoxRound(counter [4]uint32, key [2]uint32) [4]uint32 {
	product0 := uint64(philoxM0) * uint64(counter[0])
	product1 := uint64(philoxM1) * uint64(counter[2])
	return [4]uint32{
		uint32(product1>>32) ^ counter[1] ^ key[0],
		uint32(product1),
		uint32(product0>>32) ^ counter[3] ^ key[1],
		uint32(product0),
	}
}

// ReferenceNormal fills dst with deterministic normal samples using the same
// counter mapping and Box-Muller pairing as the Metal kernel. Transcendental
// rounding can differ slightly between Go and an Apple GPU; Philox's integer
// output is the cross-device bit-exact contract.
func ReferenceNormal(dst []float32, seed, counter uint64, mean, stddev float32) {
	for block := uint64(0); block*4 < uint64(len(dst)); block++ {
		current := counter + block
		random := Philox4x32(
			[4]uint32{uint32(current), uint32(current >> 32), 0, 0},
			[2]uint32{uint32(seed), uint32(seed >> 32)},
		)
		values := boxMuller4(random)
		base := int(block * 4)
		for lane := 0; lane < 4 && base+lane < len(dst); lane++ {
			dst[base+lane] = mean + stddev*values[lane]
		}
	}
}

func boxMuller4(random [4]uint32) [4]float32 {
	uniform := [4]float64{}
	for i, value := range random {
		uniform[i] = uniformOpen(value)
	}

	r0 := math.Sqrt(-2 * math.Log(uniform[0]))
	t0 := 2 * math.Pi * uniform[1]
	r1 := math.Sqrt(-2 * math.Log(uniform[2]))
	t1 := 2 * math.Pi * uniform[3]
	return [4]float32{
		float32(r0 * math.Cos(t0)),
		float32(r0 * math.Sin(t0)),
		float32(r1 * math.Cos(t1)),
		float32(r1 * math.Sin(t1)),
	}
}

func uniformOpen(value uint32) float64 {
	const float23Unit = 1.0 / 8388608.0
	return (float64(value>>9) + 0.5) * float23Unit
}
