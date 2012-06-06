package mm

func DefaultBufSize() int { return N / warp }

type Box interface{}

func Connect(dst Box, dstChan string, src Box, srcChan string) {
}
