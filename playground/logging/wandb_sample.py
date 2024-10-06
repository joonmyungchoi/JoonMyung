import argparse, torch, os, wandb, math
import torch.nn.functional as F
from joonmyung import analysis

test_acc, best_acc, optimizer = 0, 0, 0 # Dummy
model, images, labels, outputs, predicted, test_table, log_counter, epoch = None, None, None, None, None, None, None, None




parser = argparse.ArgumentParser(description='PyTorch Wandb Sapmle')
parser.add_argument('--use_wandb', action='store_true')
parser.add_argument('--wandb_project', default='test', type=str)
# parser.add_argument('--wandb_project', default='2023ICCV', type=str)
parser.add_argument('--wandb_entity', default='joonmyung', type=str)
parser.add_argument('--wandb_table', default='Sample', type=str)
parser.add_argument('--wandb_version', default='1.0.0', type=str, help="version")
parser.add_argument('--output_dir', default='2023ICCV', type=str)
parser.add_argument('--save_every', default=None, type=int, help='save model every epochs')
parser.add_argument('--server', default='154', type=str)
parser.add_argument('--paper', default='patchup', type=str)


args = parser.parse_args()
# 1-A. Wandb Log Init
if analysis.is_main_process() and args.use_wandb:
    import wandb
    wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_table)
    if not args.output_dir: args.output_dir = wandb.run.dir
    wandb.watch(model, log='all')
    wandb.config.update(args)
    torch.save({'args': args, }, os.path.join(wandb.run.dir, "args.pt"))


# 1-B. Wandb Log
if args.use_wandb:
    log = {"acc1": test_acc,
           "best_acc1": best_acc,
           "lr": optimizer.param_groups[0]["lr"]}
    wandb.log(log)


# 2-A. Wandb Table Init
columns = ["id", "image", "prediction", "label", *["logit_{}".format(d) for d in range(10)]]
test_table = wandb.Table(columns=columns)

# 2-B. Wandb Table
def log_test_predictions(images, labels, outputs, predicted, test_table, epoch, NUM_IMAGES_PER_BATCH=math.inf):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(epoch)
        test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break

log_test_predictions(images, labels, outputs, predicted, test_table, epoch)