Set-Location "d:\Claude-Code-R\MuMax-CO\mumax3-src\test"
$resultsDir = "_results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
$summary = "$resultsDir/summary.txt"
Remove-Item $summary -ErrorAction SilentlyContinue
$mx3files = Get-ChildItem "*.mx3" | Sort-Object Name
$i = 0
foreach ($f in $mx3files) {
    $i++
    $name = $f.BaseName
    $outdir = "$resultsDir/$name.out"
    Remove-Item -Recurse -Force $outdir -ErrorAction SilentlyContinue
    $logfile = "$resultsDir/$name.log"
    $errfile = "$resultsDir/$name.err"
    $p = Start-Process -FilePath "..\mumax3.exe" -ArgumentList "-o", $outdir, $f.FullName -NoNewWindow -PassThru -RedirectStandardOutput $logfile -RedirectStandardError $errfile
    $finished = $p.WaitForExit(120000)
    if (-not $finished) {
        try { $p.Kill() } catch {}
        Add-Content $summary "$name : TIMEOUT"
    } else {
        $p.Refresh()
        $code = $p.ExitCode
        if ($code -eq 0) {
            Add-Content $summary "$name : PASS"
        } else {
            Add-Content $summary "$name : FAIL (exit $code)"
        }
    }
    Write-Output "[$i/$($mx3files.Count)] $name done"
}
Add-Content $summary "ALL DONE"
