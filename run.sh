gpu=0

model_name=LightGCN_NT
model_type=graph

item_ranking=10,20,40
embedding_size=64
epoch=200
batch_size=2048
learning_rate=0.001
reg_lambda=0.0001
n_layer=2

loss_type=ssm
tau=0.2

dataset=ml-1m

alpha_uu=1.2
alpha_ii=0.8
alpha_ui=0.8
alpha_iu=1.0

python main.py \
    --gpu $gpu \
    --dataset $dataset \
    --model_name $model_name \
    --model_type $model_type \
    --item_ranking $item_ranking \
    --embedding_size $embedding_size \
    --epoch $epoch \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --reg_lambda $reg_lambda \
    --n_layer $n_layer \
    --alpha_uu $alpha_uu \
    --alpha_ii $alpha_ii \
    --alpha_ui $alpha_ui \
    --alpha_iu $alpha_iu \
    --loss_type $loss_type \
    --tau $tau
