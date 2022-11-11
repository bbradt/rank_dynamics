# Experiment 3: BIGGER MODEL
# Testing a low-dimensional MultivariateGaussian, embedded in a higher-dimensional space
# Adding more classes, with fewer samples per class
root_dir='experiments/beta_ae'
expname='beta_ae'
ID='E004'
grid_file=$root_dir'/'$expname'_'$ID'.grid'
rm $grid_file;
lrs=( '1e-3' )
optimizers=( '"adam"' )
models=( '"linear_network"' )
model_args=( '[128,[128,128,128],128]' )
model_kwargs=( '{"output_activation":nn.Softmax}' )
datasets=( '"random_mixture"' )
dataset_args=( '[]' )
# Data set kwargs
rm_modes=( '10' )
rm_Ns=( '256' )
rm_args=( '[0.5, 0.5]' )
rm_kwargs=( '{}' )
rm_distributions=( '"Beta"' )
rm_source_dims=( 2 )
rm_embedding_dims=( 128 )
rm_loc_methods=( '"rand"' )
rm_loc_args=( '[]' )
rm_cov_methods=( '"rand"' )
rm_cov_args=( '[]' )
rm_gen_methods=( 'None' )
rm_gen_args=( 'None' )
batch_sizes=( 128 )
seeds=( 314159 )
epochs=( 10 )
ks=( 0 1 2 3 4 )
criterions=( '"mse"' )
iii=0
for lr in "${lrs[@]}"; do
for optimizer in "${optimizers[@]}"; do
for model in "${models[@]}"; do
for model_arg in "${model_args[@]}"; do
for model_kwarg in "${model_kwargs[@]}"; do
for dataset in "${datasets[@]}"; do
for batch_size in "${batch_sizes[@]}"; do
for seed in "${seeds[@]}"; do 
for epoch in "${epochs[@]}"; do
for criterion in "${criterions[@]}"; do
for rm_mode in "${rm_modes[@]}"; do
for rm_N in "${rm_Ns[@]}"; do
for rm_arg in "${rm_args[@]}"; do
for rm_kwarg in "${rm_kwargs[@]}"; do
for rm_distribution in "${rm_distributions[@]}"; do
for rm_source_dim in "${rm_source_dims[@]}"; do 
for rm_embedding_dim in "${rm_embedding_dims[@]}"; do
for rm_loc_method in "${rm_loc_methods[@]}"; do
for rm_loc_arg in "${rm_loc_args[@]}"; do
for rm_cov_method in "${rm_cov_methods[@]}"; do
for rm_cov_arg in "${rm_cov_args[@]}"; do
for rm_gen_method in "${rm_gen_methods[@]}"; do 
for rm_gen_arg in "${rm_gen_args[@]}"; do
for k in "${ks[@]}"; do
    full_expname=$expname"_"$ID"_k"$k"_"$iii
    dataset_kwargs='{"modes":'${rm_mode}',"N":'${rm_N}',"args":'${rm_arg}',"kwargs":'${rm_kwarg}',"distributions":'${rm_distributions}',"source_dim":'${rm_source_dim}',"embedding_dim":'${rm_embedding_dim}',"loc_methods":'${rm_loc_method}',"loc_args":'${rm_loc_arg}',"cov_methods":'${rm_cov_method}',"cov_args":'${rm_cov_arg}', "gen_methods":'${rm_gen_method}', "gen_args":'${rm_gen_arg}'}'
    #echo dataset_kwargs $dataset_kwargs
    kwargs="--lr "${lr}" --optimizer "${optimizer}" --model "${model}" --model_args '"${model_arg}"' --model_kwargs '"${model_kwarg}"' --dataset "${dataset}" --dataset_args '"${dataset_args}"' --dataset_kwargs '"${dataset_kwargs}"' --batch_size "${batch_size}" --seed "${seed}" --epochs "${epoch}" --k "${k}" --criterion "${criterion}" --name "${full_expname}    
    kwargs=$kwargs" --autoencoder True"
    iii=$(( iii + 1 ))
    echo $kwargs >> $grid_file
done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;done;