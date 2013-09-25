package main

// Output in dump format
// Author: Arne Vansteenkiste

//import (
//	"github.com/mumax/3/dump"
//	"os"
//)
//
//func dumpDump(file string, f *data.Slice) {
//	out, err := os.OpenFile(file, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
//	core.Fatal(err)
//	defer out.Close()
//
//	w := dump.NewWriter(out, dump.CRC_ENABLED)
//	w.Header = f.Header
//	w.WriteHeader()
//	w.WriteData(f.Data)
//	w.WriteHash()
//}
