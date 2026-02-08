#!/bin/bash
set -e

echo "=== Clearing pip cache ==="
pip cache purge || true

echo "=== Installing packages ==="
pip install --no-cache-dir --force-reinstall Flask==3.0.0 flask-cors==4.0.0 gunicorn==21.2.0
pip install --no-cache-dir --force-reinstall praw==7.7.1
pip install --no-cache-dir --force-reinstall "pandas<2.1" "numpy<1.25"
pip install --no-cache-dir --force-reinstall nltk==3.8.1

echo "=== Installing OpenAI with compatible httpx ==="
pip uninstall -y openai httpx || true
pip install --no-cache-dir httpx==0.27.2
pip install --no-cache-dir openai==1.54.0

echo "=== Verifying installations ==="
python -c "import flask; print('Flask OK')"
python -c "import nltk; print('NLTK OK')"
python -c "import httpx; print('httpx version:', httpx.__version__)"
python -c "import openai; print('OpenAI version:', openai.__version__)"

echo "=== Build complete ==="
