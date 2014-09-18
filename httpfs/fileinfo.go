package httpfs

//import (
//	"os"
//	"time"
//)
//
//type fileInfo struct {
//	Nm   string      // base name of the file
//	Sz   int64       // length in bytes for regular files; system-dependent for others
//	Md   os.FileMode // file mode bits
//	MdTm time.Time   // modification time
//}
//
//func (f *fileInfo) Name() string       { return f.Nm }
//func (f *fileInfo) Size() int64        { return f.Sz }
//func (f *fileInfo) Mode() os.FileMode  { return f.Md }
//func (f *fileInfo) ModTime() time.Time { return f.MdTm }
//func (f *fileInfo) IsDir() bool        { return f.Mode().IsDir() }
//func (f *fileInfo) Sys() interface{}   { return nil }
