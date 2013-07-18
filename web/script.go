package web

// Handlers for web script input "script/"

//import (
//	"code.google.com/p/mx3/engine"
//	"code.google.com/p/mx3/util"
//	"encoding/json"
//	"fmt"
//	"log"
//	"net/http"
//)
//
//type Resp struct {
//	Err string
//	Val interface{}
//}
//
//func scriptHandler(w http.ResponseWriter, r *http.Request) {
//	cmd := r.URL.Path[len("/script/"):]
//
//	var resp Resp // TODO
//
//	defer func() {
//		bytes, err2 := json.Marshal(resp)
//		util.FatalErr(err2)
//		log.Println("resp:", string(bytes))
//		w.Write(bytes)
//	}()
//
//	code, err := engine.Compile(cmd)
//	if err != nil {
//		resp.Err = err.Error()
//		util.LogErr(err)
//		return
//	}
//
//	engine.Inject <- func() {
//		defer func() {
//			if err := recover(); err != nil {
//				resp.Err = fmt.Sprint(err)
//			}
//		}()
//		log.Println("exec:", code)
//		code.Eval()
//	}
//}
