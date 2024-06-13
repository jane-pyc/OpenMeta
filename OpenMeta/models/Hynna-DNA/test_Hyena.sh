export http_proxy="http://www.zangzelin.fun:4081"
export https_proxy="http://www.zangzelin.fun:4081"
export all_proxy="socks5://www.zangzelin.fun:4082"
CUDA_VISIBLE_DEVICES=5 python huggingface_tiny.py

unset http_proxy
unset https_proxy
unset all_proxy