#! Run deploy/deploy_windows.ps1 first! This generates the following executable:
$MUMAX = "$env:GOPATH\bin\mumax3.exe"

# Enter the test directory to (re)compile the cuda kernels
Set-Location ../test
    # $mumaxfiles = Get-ChildItem -include ("*.mx3", "*.go") -Name
    $mumaxfiles = Get-ChildItem -filter "*.mx3" -Name
    & $MUMAX -vet $mumaxfiles
    & $MUMAX -paranoid=false -failfast -cache=/tmp -http="" -f=true $mumaxfiles