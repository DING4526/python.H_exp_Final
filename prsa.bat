@echo off

rem Set PYTHONPATH to project root
set PYTHONPATH=D:\丁益三\大学\Python文件\SparseTSF-final

rem Set model name and other paths
set model_name=SparseTSF
set root_path_name=.\dataset
set data_path_name=PRSA.csv
set model_id_name=PRSA
set data_name=PRSA
set seq_len=720

rem Loop over different prediction lengths and run the python command for each
for %%p in (720) do (
    echo predict length=%%p
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features S ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --period_len 24 ^
        --model_type mlp ^
        --d_model 128 ^
        --enc_in 1 ^
        --train_epochs 30 ^
        --patience 5 ^
        --itr 1 ^
        --batch_size 256 ^
        --target "pm2.5" ^
        --learning_rate 0.002
)

pause
