//+build ignore

package main

import (
	. "."
	"fmt"
	"net/http"
	"time"
)

func main() {
	doc := NewDoc("/", testtempl)

	hitcount := 0
	doc.Elem("e_hitme").OnClick(func() {
		hitcount++
		doc.SetValue("e_hitcount", hitcount)
	})

	doc.Elem("e_namebox").OnChange(func() {
		doc.SetValue("e_greet",
			fmt.Sprint("Hello ", doc.Elem("e_namebox").Value(), "!"))
	})

	doc.Elem("e_check").OnChange(func() {
		name := doc.Value("e_namebox").(string)
		cookie := doc.Elem("e_cookies")
		if doc.Value("e_check") == true {
			cookie.SetValue(name + " likes cookies")
		} else {
			cookie.SetValue(name + " doesn't like cookies")
		}
	})

	doc.Elem("e_range").OnChange(func() {
		doc.SetValue("e_age", fmt.Sprint(
			doc.Value("e_namebox"), " is ",
			doc.Value("e_range"), " years old."))
	})

	go func() {
		for {
			time.Sleep(1 * time.Second)
			doc.SetValue("e_time", time.Now().Format("15:04:05"))
		}
	}()

	err := http.ListenAndServe(":7070", nil)
	if err != nil {
		panic(err)
	}

}

const testtempl = `
<html>

<head>
	<style type="text/css">
		body      { margin: 20px; font-family: Ubuntu, Arial, sans-serif; }
		hr        { border-style: none; border-top: 1px solid #CCCCCC; }
		.ErrorBox { color: red; font-weight: bold; } 
		.TextBox  { border:solid; border-color:#BBBBBB; border-width:1px; padding-left:4px;}
	</style>
	{{.JS}}
</head>

<body>

	<h1> GUI test </h1>
	<p> {{.ErrorBox}} </p>
	<hr/>

	{{.Span "e_time" "time flies..."}} <br/><br/>


	What's your name: {{.TextBox "e_namebox" ""}} &nbsp;
	{{.Span "e_greet"}} <br/><br/>

	You hit me {{.Span "e_hitcount" "0"}} times. {{.Button "e_hitme" "Hit me baby one more time!"}} <br/><br/>

	{{.CheckBox "e_check" "Like cookies?" false}} &nbsp;
	{{.Span "e_cookies" ""}} <br/><br/>


	Your age: {{.Range "e_range" 0 100 18}}
	{{.Span "e_age" ""}} <br/><br/>

	<hr/>

</body>
</html>
`
