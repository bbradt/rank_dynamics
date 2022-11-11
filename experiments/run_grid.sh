#!/bin/bash
N=`wc -l $1 | awk '{ print $1 }'`
N=$((N - 1))
grid_file=$1
echo grid_file $grid_file $N
sbatch <<-EOT
#!/bin/sh
#SBATCH --nodes=1
#SBATCH -J $2
#SBATCH --output=slurm/$2_%a.out
#SBATCH --error=slurm/$2_%a.err
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH -p qTRDGPUH,qTRDGPUM,qTRDGPUL,qTRDGPU
#SBATCH --oversubscribe
#SBATCH --time 24:00:00
#SBATCH --array=0-$N
pwd
eval "\$(conda shell.bash hook)"
id=\$((SLURM_ARRAY_TASK_ID+1))
echo id \$id
kwargs=\$(sed "\${id}q;d" $grid_file)
echo kwargs \$kwargs
conda activate torch_latest
cd /data/users2/bbaker/projects/first_dynamics
cmd="PYTHONPATH=. python experiments/main.py "\$kwargs
echo CMD \$cmd
eval \$cmd
exit 0
EOT