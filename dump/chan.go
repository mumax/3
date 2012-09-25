package dump

import "nimble-cube/core"

func Autosave(fname string, data core.RChan3, every int) {

	N := core.Prod(data.Size())
	out := NewWriter(core.OpenFile(fname), CRC_ENABLED)
	out.Components = 3 // TODO !!
	out.MeshSize = data.Size()

	for i := 0; ; i++ {
		output := data.ReadNext(N) // TODO
		if i%every == 0 {
			i = 0
			core.Debug("dump")
			out.WriteHeader()
			out.WriteData(output[0])
			out.WriteData(output[1])
			out.WriteData(output[2])
			out.WriteHash()
		}
		data.ReadDone()
	}
}
