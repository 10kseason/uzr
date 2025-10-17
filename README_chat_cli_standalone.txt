Chat CLI (standalone, single file) with optional **online adapter** updates.

Run (stable preset):
  python chat_cli_standalone.py --device cuda --ckpt uzr_scaled_100.pt     --identity "루리아" --inner_steps 5 --lam 0.003 --alpha 0.3 --prox 0.001     --max_items 20000 --tone polite --lang ko --save_transcript chat_log.json

Run (online weights enabled):
  python chat_cli_standalone.py --device cuda --ckpt uzr_scaled_100.pt --identity "루리아" --inner_steps 2500 --lam 0.003 --alpha 0.3 --prox 0.001 --max_items 20000 --tone polite --lang ko     --online --online_lr 3e-4 --online_steps 1 --replay_k 8 --conf_thresh 0.6 --revert_tol 0.5



!python infer_longrun_scaled.py --ckpt uzr_scaled_100.pt --turns 300 --inner_steps 24 --inner_lr 1.5 --max_len 96 --summary_every 50 --amp 1 --fp32_inner 1 --topk 2 --z_noise 0.05 --debug 1 --compile 1 --mark_step 1 --debug 1


python infer_longrun_scal2.py --ckpt uzr_scaled_100.pt --turns 300 --inner_steps 24 --inner_lr 1.0 --lr_boost 1.55 --z_in_dim 768 --film_gain 1.55 --alpha 0.2 --decay 0.02 --prox 1e-3 --lam 2e-3 --zslow_l2max 50 --topk 2 --fp32_inner 1 --compile 0 --amp 1 --charspace 1 --debug 1