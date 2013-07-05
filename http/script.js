
// update rate
var tick = 1000;

// show error in document
function showErr(err){
	document.getElementById("Errorbox").innerHTML = err;
}

function softRefresh(){
	showErr("");
	var response;
	try{
		var xmlHttp = new XMLHttpRequest();
		xmlHttp.open("POST", "/refresh/", false);
		xmlHttp.timeout = tick;
		xmlHttp.send(null);
		response = JSON.parse(xmlHttp.responseText);	
		for(var i=0; i<response.length; i++){
			var r = response[i];
			document.getElementById(r.ID).innerHTML = r.HTML;
		}
	}catch(e){
		showErr(e);
	}
}

setInterval(softRefresh, tick);
