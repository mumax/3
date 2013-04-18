package engine

import (
	"code.google.com/p/mx3/util"
	"html/template"
	"net/http"
)

var disptempl = template.Must(template.New("disp").Parse(dispText))

func disp(w http.ResponseWriter, r *http.Request) {
	ui.Lock()
	defer ui.Unlock()
	util.FatalErr(disptempl.Execute(w, ui))
}

const dispText = `
<!DOCTYPE html>
<html>

<body>

<img src="/render/m"/>

</body>
</html>
`
