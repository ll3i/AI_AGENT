@echo off
echo ===== GPU 환경 설치 시작 =====

echo.
echo [1/3] PyTorch (CUDA 12.4) 설치 중...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [2/3] Transformers / BitsAndBytes / Accelerate 설치 중...
pip install "transformers>=4.49.0" accelerate "bitsandbytes>=0.43.0"

echo.
echo [3/3] Qwen VL utils 설치 중...
pip install qwen-vl-utils

echo.
echo ===== 설치 완료 =====
echo.
echo 이제 run_solution_gpu.py 실행:
echo   python run_solution_gpu.py
echo.
pause
