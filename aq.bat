@echo off

rem Set PYTHONPATH to project root
set PYTHONPATH=D:\丁益三\大学\Python文件\SparseTSF-final

rem Set model name and other paths
set model_name=SparseTSF
set root_path_name=.\dataset
set data_path_name=AQ.csv
set model_id_name=AQ
set data_name=AQ
set seq_len=120

rem Loop over different prediction lengths and run the python command for each
for %%p in (192) do (
    for %%q in (0.5) do (
        echo Predict length=%%p
        python -u run_longExp.py ^
            --is_training 1 ^
            --root_path %root_path_name% ^
            --data_path %data_path_name% ^
            --model_id %model_id_name%_%seq_len%_%%p_%%q ^
            --model %model_name% ^
            --data %data_name% ^
            --features M ^
            --seq_len %seq_len% ^
            --pred_len %%p ^
            --period_len 24 ^
            --model_type mlp ^
            --d_model 128 ^
            --enc_in 9 ^
            --train_epochs 30 ^
            --patience 5 ^
            --itr 1 ^
            --batch_size 256 ^
            --target "PT08.S1(CO)" ^
            --learning_rate 0.002 ^
            --dropout_rate %%q ^
            --loss mse ^
            --test_op "7_time_encoding_2+optimizer+256_128_0.5_0.002"
    )
)
pause
