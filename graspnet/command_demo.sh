export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/log_rs/checkpoint-rs.tar
