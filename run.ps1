param([Parameter(ValueFromRemainingArguments = $true)][string[]]$BotArgs)
& "$PSScriptRoot\.venv\Scripts\python.exe" -m jumpbot live @BotArgs
exit $LASTEXITCODE
