
Write-Host "=========================================="
Write-Host "   MERS Visual Module Pipeline Runner     "
Write-Host "=========================================="

# 1. Check for PyTorch
Write-Host "`n[1/3] Checking PyTorch Environment..."
try {
    $torch_ver = python -c "import torch; print(torch.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " - Found PyTorch: $torch_ver"
        $cuda_avail = python -c "import torch; print(torch.cuda.is_available())" 2>$null
        Write-Host " - CUDA Available: $cuda_avail"
    } else {
        Write-Host " - ERROR: PyTorch not found. Please wait for the installation terminal to finish." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host " - ERROR: Python check failed." -ForegroundColor Red
    exit 1
}

# 2. Train Model
$model_path = "assets\models\visual_efficientnet.pth"
if (Test-Path $model_path) {
    Write-Host "`n[2/3] Model found at $model_path. Skipping Training." -ForegroundColor Yellow
    Write-Host " - To force retraining, delete this file and run again."
} else {
    Write-Host "`n[2/3] Starting Training (EfficientNet-B0 on FER2013)..."
    Write-Host " - This may take 5-10 minutes depending on GPU."
    python scripts\training\train_visual_cnn.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host " - Training FAILED. See errors above." -ForegroundColor Red
        exit 1
    }
    Write-Host " - Training Completed Successfully." -ForegroundColor Green
}

# 3. Evaluate Model
Write-Host "`n[3/3] Running Evaluation & Validation..."
python scripts\evaluation\evaluate_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host " - Evaluation FAILED." -ForegroundColor Red
    exit 1
}

Write-Host "`n=========================================="
Write-Host "   PIPELINE FINISHED SUCCESSFULLY       "
Write-Host "   Run 'python main.py' to launch MERS   "
Write-Host "=========================================="
