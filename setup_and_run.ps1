# Banner Generator API - Setup and Run Script
# Run this script to install dependencies and start the server

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Banner Generator API - Setup & Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "Found: $pythonVersion" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created!" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red
    exit 1
}

Write-Host "Dependencies installed successfully!" -ForegroundColor Green

# Check for Replicate API token
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($env:REPLICATE_API_TOKEN) {
    Write-Host "Replicate API token found" -ForegroundColor Green
    Write-Host "  AI inpainting will be enabled" -ForegroundColor Gray
} else {
    Write-Host "Replicate API token not set" -ForegroundColor Yellow
    Write-Host "  AI inpainting will be disabled (fallback to edge replication)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To enable AI inpainting:" -ForegroundColor Yellow
    Write-Host "  1. Get token from: https://replicate.com/account/api-tokens" -ForegroundColor Gray
    Write-Host "  2. Set environment variable: REPLICATE_API_TOKEN=your_token_here" -ForegroundColor Gray
    Write-Host "  3. Or create .env file with: REPLICATE_API_TOKEN=your_token_here" -ForegroundColor Gray
}

# Check for existing jobs
Write-Host ""
$jobCount = (Get-ChildItem -Path "storage\jobs" -Directory -ErrorAction SilentlyContinue | Measure-Object).Count
if ($jobCount -gt 0) {
    Write-Host "Found $jobCount existing test jobs in storage/jobs/" -ForegroundColor Green
} else {
    Write-Host "No existing jobs found (will be created when you submit jobs)" -ForegroundColor Gray
}

# Start server
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will start at: http://localhost:8000" -ForegroundColor Green
Write-Host "API docs available at: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn
uvicorn app.main:app --reload
