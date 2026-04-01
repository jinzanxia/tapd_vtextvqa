python infer_codes/qwen.py --gt-json path/datasets/M4-ViteVQA/Annotations/ViteVQA_0.0.2_t1s1val.json \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct/ \
    --vts-config ./GoMatching/configs/GoMatching_PP_BOVText_vit.yaml \
    --vts-model ./GoMatching/models/GoMatching_pp_vitaeb_bovtext.pth \
    --video-dir path/datasets/M4-ViteVQA/video/ \
    --output ./results/mot.json