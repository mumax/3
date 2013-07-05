
function http(method, url){
	var xmlHttp = new XMLHttpRequest();
	xmlHttp.open(method, url, false);
	xmlHttp.send(null);
	return xmlHttp.responseText;
}

function softRefresh(){
	try{
		var bytes = http("POST", "/refresh/");
		var response = JSON.parse(bytes);	
		if (response.Err != null){
			alert(response.Err);
		}
	}catch(e){
		alert(e);
	}
	// ...
}

setInterval(softRefresh, 1000);
