
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
from torch.nn import functional as F



def plot_xgb(evals_result, epoch, model):
    plt.plot(evals_result['train']['mlogloss'], label='Train Loss')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Loss Curve')
    plt.legend()
    plt.savefig(f"XGB_{epoch}")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=30, importance_type='weight', ax=ax)
    plt.savefig(f"XGB_important_{epoch}_Top 20 features")
    plt.close()




# plot_xgb(evals_result, epoch, model_xgb)


def run_xgb(diverse_pred_traj, future_traj):
    BN = diverse_pred_traj.shape[0]
    S = diverse_pred_traj.shape[1]
    T = diverse_pred_traj.shape[2]

    target_expanded = future_traj.reshape(BN,T, 2).unsqueeze(1).expand_as(diverse_pred_traj)
    # print("future_traj", future_traj.shape)

    target_flattened = target_expanded.reshape(BN, S, T * 2)

    target_flattened = diverse_pred_traj.reshape(BN, S * T * 2)  # (B*N, S*T*2)
    # print("target_flattened", target_flattened)
    target_concat = target_flattened.unsqueeze(1).expand(BN, S, S * T * 2).reshape(BN * S, S * T * 2)  # (B*N S, T*2*S)
    # print("target_concat", target_concat)
    # serial_numbers = torch.arange(S).repeat(BN).unsqueeze(-1).cuda()  # (B*N, S)
    serial_numbers = torch.arange(S).repeat(BN).unsqueeze(-1)# (B*N, S)

    final_output = torch.cat((target_concat, serial_numbers), dim=-1)  # (B*N*S, T*2*S+ 1)

    diff = diverse_pred_traj - target_expanded  # Difference
    dist_squared = diff.pow(2).sum(dim=-1).sum(dim=-1)  # Sum squared differences across T and 2 dimensions
    soft_targets = F.softmax(-dist_squared, dim=1)
    ranking = torch.argsort(dist_squared, dim=-1)  # (B*N, S)
    class_labels = torch.argsort(ranking, dim=-1)
    # class_labels_expanded = class_labels.repeat_interleave(S, dim=0) #BNS,S

    X_train = final_output.cpu().detach().numpy()
    y_train = class_labels.cpu().numpy().reshape(-1)  # BNS


    # evals_result = {}
    # params = {
    #     'objective': 'multi:softmax',
    #     'num_class': 20,
    #     'eval_metric': 'mlogloss',
    #     'max_depth': 4,
    #     'learning_rate': 0.1,
    #     'device': 'cuda',
    # }
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # if iter_num == 0:
    #     new_model = xgb.train(params=params, dtrain=dtrain, evals=[(dtrain, "train")], evals_result=evals_result,
    #                           verbose_eval=False)
    # else:
    #     new_model = xgb.train(params=params, dtrain=dtrain, evals=[(dtrain, "train")], evals_result=evals_result,
    #                           xgb_model=xgb_model, verbose_eval=False)

    return X_train, y_train