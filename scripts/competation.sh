export CUDA_VISIBLE_DEVICES=1
python -u ../run.py \
  --root_path ../dataset/ \
  --data_path  APS-PV.csv \
  --data custom \
  --freq t \
  --features MS \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 24 \
  --c_in 17 \
  --c_out 17 \
  --input 17 \
  --batch_size 64 \
  --patience 10 \
  --use_gpu True \