# Start LiteLLM Proxy Server (Windows)

Write-Host "Starting LiteLLM Proxy Server..." -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Load .env file
$envFile = ".env"
if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile..." -ForegroundColor Yellow
    Get-Content $envFile | ForEach-Object {
        if ($_ -notmatch '^\s*#' -and $_.Trim() -ne '') {
            $key, $value = $_.Split('=', 2)
            [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), 'Process')
        }
    }
    Write-Host "`u{2713} Loaded .env file" -ForegroundColor Green
}

# Verify NVIDIA_API_KEY is set
$nvKey = [Environment]::GetEnvironmentVariable('NVIDIA_API_KEY', 'Process')
if ([string]::IsNullOrEmpty($nvKey)) {
    Write-Host "`u{274C} Error: NVIDIA_API_KEY is not set" -ForegroundColor Red
    exit 1
}

Write-Host "`u{2713} NVIDIA_API_KEY is set" -ForegroundColor Green

# Start LiteLLM proxy on port 4000
Write-Host "Starting proxy on port 4000..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

python -m litellm.proxy.proxy_cli `
    --config litellm_config.yaml `
    --port 4000 `
    --debug
