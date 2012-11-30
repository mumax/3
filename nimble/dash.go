package nimble

import(
	"fmt"
	"time"
)

var lastdash = time.Now()

func Dash(step int, t, dt, err float64){
	if time.Since(lastdash) > 100*time.Millisecond{
		lastdash = time.Now()
		fmt.Printf("step: % 8d t: % 12es Δt: % 12e ε:% 12es\u000D", step, float32(t), float32(dt), float32(err))
	}
}

func DashExit(){
	fmt.Println()
}
