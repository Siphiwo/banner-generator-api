# View output.txt log file
# Shows Replicate usage and fallback information

$outputFile = "output.txt"

if (Test-Path $outputFile) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "REPLICATE USAGE LOG" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    Get-Content $outputFile | ForEach-Object {
        if ($_ -match "✅|SUCCESS") {
            Write-Host $_ -ForegroundColor Green
        }
        elseif ($_ -match "❌|FAILED") {
            Write-Host $_ -ForegroundColor Red
        }
        elseif ($_ -match "⚠️|WARNING|FALLBACK") {
            Write-Host $_ -ForegroundColor Yellow
        }
        elseif ($_ -match "🚀|CALLING") {
            Write-Host $_ -ForegroundColor Cyan
        }
        else {
            Write-Host $_
        }
    }
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "Total entries: $((Get-Content $outputFile).Count)" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}
else {
    Write-Host "`n⚠️  output.txt not found" -ForegroundColor Yellow
    Write-Host "Start the server to generate the log file`n" -ForegroundColor Gray
}
