//+build ignore

package main

import (
	. "."
	"fmt"
	"net/http"
	"time"
)

func main() {
	data := []string{"one", "two", "three"}
	doc := NewDoc(testtempl, data)

	hitcount := 0
	doc.OnEvent("e_hitme", func() {
		hitcount++
		doc.SetValue("e_hitcount", hitcount)
	})
	doc.SetValue("e_hitme", "Hit me!")

	doc.SetValue("e_greet", "anyone there?")
	doc.OnEvent("e_namebox", func() {
		doc.SetValue("e_greet",
			fmt.Sprint("Hello ", doc.Value("e_namebox"), "!"))
	})

	doc.OnEvent("e_check", func() {
		name := doc.Value("e_namebox").(string)
		if doc.Value("e_check") == true {
			doc.SetValue("e_cookies", name+" likes cookies")
		} else {
			doc.SetValue("e_cookies", name+" doesn't like cookies")
		}
	})

	doc.OnEvent("e_range", func() {
		doc.SetValue("e_age", fmt.Sprint(
			doc.Value("e_namebox"), " is ",
			doc.Value("e_range"), " years old."))
	})

	doc.OnEvent("e_os", func() { doc.SetValue("e_os_opinion", doc.Value("e_os").(string)+", really?") })

	go func() {
		for {
			time.Sleep(1 * time.Second)
			doc.SetValue("e_time", time.Now().Format("15:04:05"))
		}
	}()

	http.Handle("/gui", doc)
	http.Handle("/gui2", doc)
	println("listening :7070")
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
	{{.Span "e_greet" ""}} <br/><br/>


	You hit me {{.Span "e_hitcount" "0"}} times. {{.Button "e_hitme" "Hit me baby one more time!"}} <br/><br/>


	{{.CheckBox "e_check" "Like cookies?" false}} &nbsp;
	{{.Span "e_cookies" ""}} <br/><br/>


	Your age: {{.Range "e_range" 0 100 18}}
	{{.Span "e_age" ""}} <br/><br/>


	Favorite OS: {{.BeginSelect "e_os"}}
		{{.Option "Windows"}}
		{{.Option "MacOSX" }}
		{{.Option "Linux"  }}
		{{.Option "BSD"    }}
		{{.Option "Minix"  }}
		{{.Option "Plan9"  }}
		{{.Option "other"  }}
	{{.EndSelect}}

<br/>
	{{.Span "e_os_opinion" " "}}
	<br/> <br/>


	{{range .Data}}
	{{$.Button . . }}
	{{end}}


	<hr/>

</body>
</html>
`
