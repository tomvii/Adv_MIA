# cls nat
python trainCls.py --dataset mesidorBin --wd 1e-3 --lr 0.01 --method nat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset mesidorMulti --wd 1e-3 --lr 0.01 --method nat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melBin --wd 1e-3 --lr 0.01 --method nat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melMulti --wd 1e-3 --lr 0.01 --method nat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset xrayBin --wd 1e-3 --lr 0.01 --method nat --model chexNet --eps 0 --epoch 100 --bs 32 --gpu 7
# cls pgdat
python trainCls.py --dataset mesidorBin --wd 1e-3 --lr 0.01 --method pgdat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset mesidorMulti --wd 1e-3 --lr 0.01 --method pgdat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melBin --wd 1e-3 --lr 0.01 --method pgdat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melMulti --wd 1e-3 --lr 0.01 --method pgdat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset xrayBin --wd 1e-3 --lr 0.01 --method pgdat --model chexNet --eps 0 --epoch 100 --bs 32 --gpu 7
# cls mart
python trainCls.py --dataset mesidorBin --wd 1e-3 --lr 0.01 --method mart --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset mesidorMulti --wd 1e-3 --lr 0.01 --method mart --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melBin --wd 1e-3 --lr 0.01 --method mart --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melMulti --wd 1e-3 --lr 0.01 --method mart --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset xrayBin --wd 1e-3 --lr 0.01 --method mart --model chexNet --eps 0 --epoch 100 --bs 32 --gpu 7
# cls trades
python trainCls.py --dataset mesidorBin --wd 1e-3 --lr 0.01 --method trades --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset mesidorMulti --wd 1e-3 --lr 0.01 --method trades --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melBin --wd 1e-3 --lr 0.01 --method trades --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melMulti --wd 1e-3 --lr 0.01 --method trades --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset xrayBin --wd 1e-3 --lr 0.01 --method trades --model chexNet --eps 0 --epoch 100 --bs 32 --gpu 7
# cls hat
python trainCls.py --dataset mesidorBin --wd 1e-3 --lr 0.01 --method hat --model res18 --helperModel "res18, chexnet, or mv2" --helperWeight "path/to/nat/weight" --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset mesidorMulti --wd 1e-3 --lr 0.01 --method hat --model res18 --helperModel "res18, chexnet, or mv2" --helperWeight "path/to/nat/weight" --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melBin --wd 1e-3 --lr 0.01 --method hat --model res18 --helperModel "res18, chexnet, or mv2" --helperWeight "path/to/nat/weight" --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset melMulti --wd 1e-3 --lr 0.01 --method hat --model res18 --helperModel "res18, chexnet, or mv2" --helperWeight "path/to/nat/weight" --eps 0 --epoch 100 --bs 32 --gpu 7
python trainCls.py --dataset xrayBin --wd 1e-3 --lr 0.01 --method hat --model chexNet  --helperModel "res18, chexnet, or mv2" --helperWeight "path/to/nat/weight" --eps 0 --epoch 100 --bs 32 --gpu 7
# multi label cls
python trainChex.py --wd 1e-3 --lr 0.01 --method nat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
python trainChex.py --wd 1e-3 --lr 0.01 --method pgdat --model res18 --eps 0 --epoch 100 --bs 32 --gpu 7
# seg nat
python trainSeg.py --dataset mel --wd 1e-3 --lr 1e-4 --method nat --model unet --eps 0 --epoch 100 --bs 32 --gpu 7
python trainSeg.py --dataset xray --wd 1e-3 --lr 1e-4 --method nat --model unet --eps 0 --epoch 100 --bs 32 --gpu 7
# seg pgdat
python trainSeg.py --dataset mel --wd 1e-3 --lr 1e-4 --method pgdat --model unet --eps 0 --epoch 100 --bs 32 --gpu 7
python trainSeg.py --dataset xray --wd 1e-3 --lr 1e-4 --method pgdat --model unet --eps 0 --epoch 100 --bs 32 --gpu 7