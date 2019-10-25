from src import *
from src.networks import *
from src.models import *
from options import *

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork(args.channels)
iw_network = ImageWiseNetwork(args.channels)

if args.network == '0' or args.network == '1':
    pw_model = PatchWiseModel(args, pw_network)
    pw_model.validate()

if args.network == '0' or args.network == '2':
    iw_model = ImageWiseModel(args, iw_network, pw_network)
    iw_model.validate(roc=False)
    #iw_model.validate_voter()
