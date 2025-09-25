

# ------------- teacher net --------------------#
main_path = './chkpnt/teachers/models/'
teacher_model_path_dict = {
    'VITB16': main_path + 'VITB16_cifar100_lr_0.05_decay_0.0005_trial_0/VITB16_best.pth',
    'VITS16': main_path + 'VITS16_cifar100_lr_0.05_decay_0.0005_trial_0/VITS16_best.pth',
    #'UNI': main_path + 'UNI_pcam_lr_0.001_decay_0.0005_trial_0/UNI_best.pth',
    #'DINOL14': main_path + 'DINOL14_pcam_lr_0.001_decay_0.0005_trial_0/DINOL14_best.pth',
    #'RegNetY_400MF': main_path + 'RegNetY_400MF_cifar100_lr_0.05_decay_0.0005_trial_0/RegNetY_400MF_best.pth',
    #'RegNetX_400MF': main_path + 'RegNetX_400MF_cifar100_lr_0.05_decay_0.0005_trial_0/RegNetX_400MF_best.pth',
    #'resnet32x4': main_path + 'resnet32x4_cifar100_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth',
    #'wrn_28_4': main_path + 'wrn_28_4_cifar100_lr_0.05_decay_0.0005_trial_0/wrn_28_4_best.pth',
    #'resnet110x2': '/data/winycg/checkpoints/mkd_checkpoints/teachers/pretrained_models/resnet110x2_best.pth',
    #'ResNet50': '/data/winycg/imagenet_pretrained/resnet50-0676ba61.pth',
    #'ResNet101': '/data/winycg/imagenet_pretrained/resnet101-63fe2227.pth',
    #'wide_resnet50_2': '/data/winycg/imagenet_pretrained/wide_resnet50_2-95faca4d.pth',
    #'resnext50_32x4d': '/data/winycg/imagenet_pretrained/resnext50_32x4d-7cdf4587.pth',
    }
