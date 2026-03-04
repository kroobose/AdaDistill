from easydict import EasyDict as edict

config = edict()

# Dataset
config.dataset = "emoreIresNet"  # training dataset

# Model
config.network = "iresnet50"  # iresnet100 | iresnet50 | iresnet18 | mobilefacenet
config.teacher = "iresnet50"
config.embedding_size = 512  # embedding size of model
config.SE = False  # SEModule
config.fp16 = True

# Optimization
config.batch_size = 256  # batch size per GPU
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# Loss
config.loss = "ArcFace"  # Option : ArcFace, CosFace, MLLoss
config.s = 64.0
config.m = 0.45
config.adaptive_alpha = True

# Paths / Resume
config.output = "output/teacher/"  # train model output folder
config.pretrained_teacher_path = "output/teacher/295672backbone.pth"
config.pretrained_teacher_header_path = "output/teacher/295672header.pth"  # teacher folder
config.global_step = 0  # step to resume

if config.dataset == "emoreIresNet":
    config.rec = "/workspace/dataset/test"
    config.data_path = "/workspace/dataset/MS1MV2/train"
    config.db_file_format = "folder"
    config.sample = 200

    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 26
    config.warmup_epoch = -1
    config.eval_step = 5686
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

    config.lr_func = lr_step_func
