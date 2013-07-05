
// update rate
var tick = 1000;

// show error in document
function showErr(err){
	document.getElementById("Errorbox").innerHTML = err;
}

// refreshes the contents of all dynamic elements,
// leaves the rest of the page alone.
function refresh(){
	showErr("");
	var response;
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/refresh/", false);
		req.timeout = tick;
		req.send(null);
		response = JSON.parse(req.responseText);	
		for(var i=0; i<response.length; i++){
			var r = response[i];
			document.getElementById(r.ID).innerHTML = r.HTML;
		}
	}catch(e){
		showErr(e);
	}
}

function rpc(method){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/rpc/", false);
		req.timeout = tick;
		var map = {"Method": method};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e);
	}
	refresh();
}

setInterval(refresh, tick);
