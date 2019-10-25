import os 
from src import *
from src.networks import *
from src.models import *
from options import *
#import send_qqmail

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork(args.channels)
iw_network = ImageWiseNetwork(args.channels)

if args.network == '0' or args.network == '1':
    pw_model = PatchWiseModel(args, pw_network)
    res = pw_model.train()
    best_epoch = res[2]
    checkpoints_path = args.checkpoints_path 
    os.system("cp " + args.checkpoints_path + "/weights_pw1.pth.e"+str(best_epoch) \
            +" " + args.checkpoints_path + "/weights_pw1.pth")
    #notify me if finish
    #send_qqmail.send_mail(subject="Training Complete", content=args.checkpoints_path)

if args.network == '0' or args.network == '2':
    iw_model = ImageWiseModel(args, iw_network, pw_network)
    res = iw_model.train()
    best_epoch = res[1]
    if (args.patches_overlap == False):
        os.system("cp " + args.checkpoints_path + "/weights_iw1.pth.e"+str(best_epoch) +" " + args.checkpoints_path + "/weights_iw1.pth")
    else:
        os.system("cp " + args.checkpoints_path + "/weights_iw1.pth.e"+str(best_epoch) +" " + args.checkpoints_path + "/weights_iw1.pth_cross_spatial")

