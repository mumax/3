package engine

import (
	"fmt"
	"log"
	"os"
)

var (
	hist    string   // console history for GUI
	logfile *os.File // saves history of input commands +  output
)

func LogIn(msg ...interface{}) {
	logGUI("", sprint(msg...), "")
	logFile(msg...)
	fmt.Println(msg...)
}

func LogOut(msg ...interface{}) {
	msg2 := "/*" + fmt.Sprintln(msg...) + "*/"
	logFile(msg2)
	fmt.Println(msg2)
}

func LogErr(msg ...interface{}) {
	msg2 := "/*" + fmt.Sprintln(msg...) + "*/"
	logFile(msg2)
	logGUI(`<b>`, msg2, `</b>`)
	fmt.Fprintln(os.Stderr, msg2)
}

func logFile(msg ...interface{}) {
	out := openlog()
	if out != nil {
		fmt.Fprintln(out, msg...)
	}
}

func logGUI(pre, msg, post string) {
	if len(msg) > 1000 {
		msg = msg[:1000-len("...")] + "..."
	}
	if hist != "" { // prepend newline
		hist += "\n"
	}
	hist += pre + msg + post
	// TODO: push to web !!
}

// returns log file of input commands, opening it first if needed
func openlog() *os.File {
	if logfile == nil {
		var err error
		logfile, err = os.OpenFile(OD+"/input.log", os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			log.Println(err)
		}
	}
	return logfile
}

// like fmt.Sprint but with spaces between args
func sprint(msg ...interface{}) string {
	str := fmt.Sprintln(msg...)
	str = str[:len(str)-1] // strip newline
	return str
}
