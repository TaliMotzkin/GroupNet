import os

import torch

def saveModel(G, D,M, args,epoch=''):
    timestamp = args.timestamp
    output_dir = "%s%s" % (args.model_dir, timestamp)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.save(G.state_dict(), "%s/G_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(D.state_dict(), "%s/D_%s_%s.pth" % (output_dir,args.dataset,epoch))
    torch.save(M.state_dict(), "%s/M_%s_%s.pth" % (output_dir, args.dataset, epoch))