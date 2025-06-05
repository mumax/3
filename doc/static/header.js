function mod(n, m) {
    return ((n % m) + m) % m
}

function xmas() { // Show xmas icon from Christmas eve until the first weekend starts after Jan 4th
    var now = new Date();
    var year = now.getFullYear();
    var xmasBegin = new Date(year, 11, 24); // December 24th 0:00:00, so day of Christmas eve
    var xmasEnd = new Date(year, 0, 5); // January 5th 0:00:00, earliest moment to end xmas
    var xmasEnd = new Date(year, 0, xmasEnd.getDate() + mod(6 - xmasEnd.getDay(), 7)); // Weekend after xmasEnd (Fri/sat midnight)
    var its_xmas = now >= xmasBegin || now <= xmasEnd;
    var xmas_img_file = its_xmas ? "nimble-cubes128-xmas.png" : "nimble-cubes128.png";
    document.getElementById("header_img").setAttribute("src", xmas_img_file);
    document.querySelector("link[rel~='icon']").href = xmas_img_file;
}
xmas()
setInterval(xmas, 5 * 60 * 1000) // Repeat every 5 minutes to update icon if page is left open