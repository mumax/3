package nc

import (
	"fmt"
)

func Panic(msg ...interface{}) {
	panic(fmt.Sprint(msg...))
}
