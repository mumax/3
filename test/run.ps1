# Runs all the .mx3 and .go files in this directory as tests
#! Run deploy/deploy_windows.ps1 first! This generates the following executable:

param ( # Optional arguments. Example usage: ./run.ps1 -MUMAX "C:\Programs\mumax\mumax3.exe"
    [String]$MUMAX = "$env:GOPATH\bin\mumax3.exe" # Path to the mumax3 executable
)

# Enter the test directory to (re)compile the cuda kernels
Set-Location ../test
    $mumaxfiles = Get-ChildItem -filter "*.mx3" -Name
    $mumaxandgofiles = Get-ChildItem -include ("*.mx3", "*.go") -Name
    & $MUMAX -vet $mumaxfiles
    & $MUMAX -paranoid=false -failfast -cache=/tmp -http="" -f=true $mumaxandgofiles