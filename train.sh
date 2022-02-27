device=0
i=1e-4
j=0.1
for size in 11
do
    for data in IP PU SV KSC
    do
        CUDA_VISIBLE_DEVICES=$device python train.py --dataset $data --epochs 500 --crf True --crf_channel 10 --weight_decay $i --mu $j --inplanes 256 --spatialsize $size
    done
done
