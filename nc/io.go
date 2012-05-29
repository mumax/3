package nc

import (
	"log"
)

func init(){
	log.SetFlags(log.Lmicroseconds|log.Lshortfile)
}

func Println(msg ...interface{}) {
	log.Println(msg...)
}
