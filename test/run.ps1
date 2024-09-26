#! Run deploy/deploy_windows.ps1 first! This generates the following executable:
$MUMAX = "$env:GOPATH\bin\mumax3.exe"

# Enter the test directory to (re)compile the cuda kernels
Set-Location ../test
    $mumaxfiles = Get-ChildItem -filter "*.mx3" -Name
    $mumaxandgofiles = Get-ChildItem -include ("*.mx3", "*.go") -Name
    & $MUMAX -vet $mumaxfiles
    & $MUMAX -paranoid=false -failfast -cache=/tmp -http="" -f=true $mumaxandgofiles