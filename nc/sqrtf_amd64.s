// func Sqrtf(x float32) float32
TEXT ·Sqrtf+0(SB),$0-16
	SQRTSS	 x+0(FP), X0
	MOVSS   X0,r+8(FP)
	RET
