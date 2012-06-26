package safe

import ()

type Float32s struct{ slice }

func MakeFloat32s(len_ int) Float32s {
	return Float32s{makeslice(len_, sizeofFloat32)}
}

func (s Float32s) Slice(start, stop int) Float32s {
	return Float32s{s.slice.slice(start, stop)}
}

const sizeofFloat32 = 4
