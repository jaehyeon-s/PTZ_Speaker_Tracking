param(
    [string]$MvpCsvPath = ""
)

if ($MvpCsvPath -ne "") {
    $env:MVPV3_CSV_PATH = $MvpCsvPath
    Write-Host "MVPV3_CSV_PATH=$env:MVPV3_CSV_PATH"
} else {
    Write-Host "MVPV3_CSV_PATH is not set. Dashboard will use runtime/logs fallback or mock data."
}

python -m uvicorn src.api.app:app --reload
