package nc

import (
	"log"
	"os"
)

func init() {
	log.SetFlags(log.Lmicroseconds  | log.Lshortfile)
	log.SetPrefix("#")
	PrintInfo(os.Stdout)
}
