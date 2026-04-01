# python -m pip install -e detectron2 --no-build-isolation
# pip install -U flash-attn==2.7.2.post1 --no-build-isolation
python setup.py build develop
TEMP_DIR=$(find build -maxdepth 1 -type d -name "temp.*" | head -n 1)
cd "$TEMP_DIR"
ninja -v
cd ../../
python setup.py build develop