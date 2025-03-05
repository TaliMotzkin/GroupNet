import argparse
import datetime
import dateutil
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="NPCGAN")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default="./out/")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="./model/")
    parser.add_argument("--dataset", dest="dataset", type=str, default="raw")
    parser.add_argument("--G_dict", dest="G_dict", type=str, default="")
    parser.add_argument("--C_dict", dest="C_dict", type=str, default="")
    parser.add_argument("--D_dict", dest="D_dict", type=str, default="")
    parser.add_argument("--traj_len", dest="traj_len", type=int, default=8)
    parser.add_argument("--delim", dest="delim", default="\t")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
    parser.add_argument("--l2_weight", type=float, default=1)
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.9)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--model_names', default=None)
    parser.add_argument('--model_save_dir', default='saved_models/fish_overlap')
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--traj_scale', type=int, default=1)
    parser.add_argument('--sample_k', type=int, default=20)
    parser.add_argument('--past_length', type=int, default=5)
    parser.add_argument('--future_length', type=int, default=10)

    parser.add_argument('--agent', default=0)
    parser.add_argument('--target', default=[80,80])
    parser.add_argument('--method', default="mean")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # if "timestamp" not in args:
    #     args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime(
    #         "%Y_%m_%d_%H_%M_%S"
    #     )
    return args


if __name__ == "__main__":
    print(parse_args())
