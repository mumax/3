package engine

import (
	"fmt"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
	"io"
	"os"
)

var (
	hist    string         // console history for GUI
	logfile io.WriteCloser // saves history of input commands +  output
)

// Special error that is not fatal when paniced on and called from GUI
// E.g.: try to set bad grid size: panic on UserErr, recover, print error, carry on.
type UserErr string

func (e UserErr) Error() string { return string(e) }

func CheckRecoverable(err error) {
	if err != nil {
		panic(UserErr(err.Error()))
	}
}

func LogIn(msg ...interface{}) {
	str := sprint(msg...)
	log2GUI(str)
	log2File(str)
	fmt.Println(str)
}

func LogOut(msg ...interface{}) {
	str := "//" + sprint(msg...)
	log2GUI(str)
	log2File(str)
	fmt.Println(str)
}

func LogErr(msg ...interface{}) {
	str := "//" + sprint(msg...)
	log2GUI(str)
	log2File(str)
	fprintln(os.Stderr, str)
}

func log2File(msg string) {
	if logfile != nil {
		fprintln(logfile, msg)
	}
}

func initLog() {
	if logfile != nil {
		panic("log already inited")
	}
	// open log file and flush what was logged before the file existed
	var err error
	logfile, err = httpfs.Create(OD() + "log.txt")
	if err != nil {
		panic(err)
	}
	util.FatalErr(err)
	logfile.Write(([]byte)(hist))
	logfile.Write([]byte{'\n'})
}

func log2GUI(msg string) {
	if len(msg) > 1000 {
		msg = msg[:1000-len("...")] + "..."
	}
	if hist != "" { // prepend newline
		hist += "\n"
	}
	hist += msg
	// TODO: push to web ?
}

// like fmt.Sprint but with spaces between args
func sprint(msg ...interface{}) string {
	str := fmt.Sprintln(msg...)
	str = str[:len(str)-1] // strip newline
	return str
}
