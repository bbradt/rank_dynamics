import os
import glob
import torch
import tqdm
import shutil
import pickle
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", default="all")
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--experiment_name", default="mvg_clf_E000")
    parser.add_argument("--output_folder", default="stats")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--folds", default=5, type=int)
    args = parser.parse_args()

    outputs_path = os.path.join(
        args.output_root, args.experiment_name, args.output_folder)
    os.makedirs(outputs_path, exist_ok=True)
    # Statistics Gathered from Hooks
    hook_directory = os.path.join(
        args.output_root, args.experiment_name, "hooks")
    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dfs = []
    target_csv = os.path.join(outputs_path,
                              "hook_stats.csv")
    print("OK TARGET ", target_csv, os.path.exists(target_csv))
    #input("press enter to continue")
    if os.path.exists(target_csv):
        os.remove(target_csv)
    print("OK TARGET ", target_csv, os.path.exists(target_csv))
    #input("press enter to continue")
    for fold in range(args.folds):
        outputs_path_ = os.path.join(
            args.output_root, args.experiment_name+"_k%d*" % (fold, ))
        hook_directory_ = os.path.join(outputs_path_, "hooks")

        print(list(glob.glob(hook_directory_)))
        print(list(glob.glob(outputs_path_)))
        hook_directory_ = list(glob.glob(hook_directory_))[0]
        outputs_path_ = list(glob.glob(outputs_path_))[0]
        os.makedirs(outputs_path_, exist_ok=True)
        for i in tqdm.tqdm(range(args.epochs)):
            #print("Starting epoch %d fold %d" % (i, fold))
            batches = glob.glob(os.path.join(hook_directory_, "e%d.pkl" % i))
            acts = dict()
            deltas = dict()
            grads = dict()
            num_batches = 0
            for hook_filename in batches:
                stats = pickle.load(open(hook_filename, "rb"))
                for k, v in stats.items():
                    if k not in acts.keys():
                        acts[k] = dict()
                        deltas[k] = dict()
                        grads[k] = dict()
                    for module in stats[k]["act"].keys():
                        if module not in acts[k].keys():
                            acts[k][module] = []
                            deltas[k][module] = []
                            grads[k][module] = []
                        acts[k][module].extend(stats[k]["act"][module])
                        num_batches = len(stats[k]["act"][module])
                        deltas[k][module].extend(stats[k]["delta"][module])
                        grads[k][module].extend(stats[k]["grad"][module])
            # total=len(list(acts.keys()))*len(list(acts[0].keys()))*num_batches
            # with tqdm.tqdm(total=total) as pbar:
            for group in acts.keys():
                #print("\tGroup %s" % group)
                for module in acts[group].keys():
                    #print("\t\tModule %s" % module)
                    for batch in range(len(acts[group][module])):
                        row = dict(epoch=i, batch=batch,
                                   group=group, module=module)
                        act = acts[group][module][batch].to(device)
                        delta = deltas[group][module][batch].to(device)
                        grad = grads[group][module][batch].to(device)
                        act_rank = torch.linalg.matrix_rank(act)
                        delta_rank = torch.linalg.matrix_rank(delta)
                        grad_rank = torch.linalg.matrix_rank(grad)
                        row['act_rank'] = act_rank.item()
                        row['delta_rank'] = delta_rank.item()
                        row['grad_rank'] = grad_rank.item()
                        row['k'] = fold
                        rows.append(row)
                        # pbar.update(1)
                df = pd.DataFrame(rows)
                df.to_csv(os.path.join(outputs_path_, "hook_stats.csv"))
        header = True if fold == 0 else False
        df.to_csv(target_csv, mode="a", header=header, index=False)
        # dfs.append(df)
    #ull_df = pd.concat(dfs).reset_index(drop=True)
    #os.makedirs(outputs_path, exist_ok=True)
    #full_df.to_csv(os.path.join(outputs_path, "hook_stats.csv"))

    print("Done")
