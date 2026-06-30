# source this to enable onnxruntime CUDA provider (torch's bundled CUDA12/cuDNN9 libs)
export LD_LIBRARY_PATH="$(python3 -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(set(os.path.dirname(p) for p in glob.glob(b+'/*/lib/*.so*')))))"):${LD_LIBRARY_PATH:-}"
