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

func logFile(msg ...interface{}) {
	out := openlog()
	if out != nil {
		fmt.Fprintln(out, msg...)
	}
}

func LogInput(msg ...interface{}) {
	logFile(msg...)
	logGUI(msg...)
}

func LogOutput(msg ...interface{}) {
	logGUI(msg...)
	msg2 := "/*" + fmt.Sprintln(msg...) + "*/"
	logFile(msg2)
}

func logGUI(msg ...interface{}) {
	m := fmt.Sprintln(msg...)
	m = m[:len(m)-1] // strip newline
	if len(m) > 1000 {
		m = m[:1000-3] + "..."
	}
	if hist != "" { // prepend newline
		hist += "\n"
	}
	hist += m
	fmt.Println(m)
}
