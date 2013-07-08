
// auto-refresh rate
var tick = 50;
var autorefresh = true;

// show error in document
function showErr(err){
	document.getElementById("ErrorBox").innerHTML = err;
}

// called on change of auto-refresh button
function setautorefresh(){
	autorefresh =  document.getElementById("AutoRefresh").checked;
}

// refreshes the contents of all dynamic elements,
// leaves the rest of the page alone.
function refresh(){
	if (autorefresh){
		showErr("");
		var response;
		try{
			var req = new XMLHttpRequest();
			req.open("POST", "/refresh/", true);
			req.timeout = tick;
			req.onreadystatechange = function(){
				if (req.readyState == 4) { // DONE
					response = JSON.parse(req.responseText);	
					for(var i=0; i<response.length; i++){
						var r = response[i];
						document.getElementById(r.ID).innerHTML = r.HTML;
					}
				}
			};
			req.send(null);
		}catch(e){
			showErr(e);
		}
	}
}

// remote procedure call, called on button clicks etc.
function rpc(method){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/rpc/", false);
		var map = {"Method": method};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e);
	}
	refresh();
}

setInterval(refresh, tick);
