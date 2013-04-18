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

<head>
</head>
<body>

<script>

	var img = new Image();
	img.src = "/render/m";

	function updateImg(){
		if(img.complete){
			document.getElementById("magnetization").src = img.src;
			img = new Image();
			img.src = "/render/m?" + new Date();
		}
	}

	setInterval(updateImg, 500);
</script>

<img id="magnetization" src="/render/m"/>

</body>
</html>
`
