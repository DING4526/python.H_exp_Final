@echo off

rem Set PYTHONPATH to project root
set PYTHONPATH=D:\丁益三\大学\Python文件\SparseTSF-final

rem Set model name and other paths
set model_name=SparseTSF
set root_path_name=.\dataset
set data_path_name=ETTh1.csv
set model_id_name=ETTh1
set data_name=ETTh1
set seq_len=720

rem Loop over different prediction lengths and run the python command for each
for %%p in (192) do (
    echo predict length=%%p
    python -u run_longExp.py ^
        --is_training 1 ^
        --root_path %root_path_name% ^
        --data_path %data_path_name% ^
        --model_id %model_id_name%_%seq_len%_%%p ^
        --model %model_name% ^
        --data %data_name% ^
        --features M ^
        --seq_len %seq_len% ^
        --pred_len %%p ^
        --period_len 24 ^
        --model_type mlp ^
        --d_model 128 ^
        --enc_in 7 ^
        --train_epochs 20 ^
        --patience 5 ^
        --itr 1 ^
        --batch_size 256 ^
        --learning_rate 0.002 ^
        --loss mse
)

pause
