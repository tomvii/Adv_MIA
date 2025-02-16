# cls
python testCls.py --dataset mesidorBin --targetModel res18 --targetWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
python testCls.py --dataset mesidorMul --targetModel res18 --targetWeight ./ckpt/ResNet_melMulti_nat_class3_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
python testCls.py --dataset melBin --targetModel res18 --targetWeight ./ckpt/ResNet_mesidorBin_nat_class2_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
python testCls.py --dataset melMul --targetModel res18 --targetWeight ./ckpt/ResNet_mesidorMulti_nat_class4_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
python testCls.py --dataset xrayBin --targetModel res18 --targetWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./ckpt/ResNet_melBin_nat_class2_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
# multi cls
python testChex.py --targetModel res18 --targetWeight ./clsCkpt/ResNet_xray_multilabel_adv0-255_0_0-255_maxLR0.1_weightDecay0.001 --surrogateModel res18 --surrogateWeight ./clsCkpt/ResNet_xray_multilabel_adv0-255_0_0-255_maxLR0.1_weightDecay0.001 --bs 16 --gpu 7
# seg
python testSeg.py --dataset xray --targetModel unet --targetWeight ./segckpt/UNet_xray_NAT_CrossEntropyLoss_Adam --surrogateModel unet --surrogateWeight ./segckpt/UNet_xray_NAT_CrossEntropyLoss_Adam --bs 16 --gpu 7
