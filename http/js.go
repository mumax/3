package main
const js = `<script>

function http(method, url){
	var xmlHttp = new XMLHttpRequest();
	xmlHttp.open(method, url, false);
	xmlHttp.send(null);
	return xmlHttp.responseText;
}

function httpGet(url){
	return http("GET", url);
}

function httpPost(url){
	return http("POST", url);
}

function rpc(command, err){
	try{
		var bytes = httpPost("/script/" + command);
		var response = JSON.parse(bytes);	
		if (response.Err != null){
			document.getElementById(err).innerHTML = response.Err;
		}
	}catch(e){
		document.getElementById(err).innerHTML = e;
	}
}

</script>`
