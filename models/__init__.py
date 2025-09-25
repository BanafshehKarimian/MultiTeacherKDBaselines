from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet110x2, resnet32x4, resnet20x4
from .resnet import resnet8x4_double
from .UNI import UNI
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_28_4, wrn_10_2, wrn_10_1 
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half, mobilenet
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2, ShuffleV2_0_5
from .DINOL14 import DINOL14
from .regnet import RegNetY_400MF, RegNetX_400MF,  RegNetX_200MF
from .policy import Policy, PolicyTrans
from .VITS8 import VITS8
from .VITS16 import VITS16
from .VITB8 import VITB8
from .VITB16 import VITB16
from .VIT import VIT

model_dict = {
    'UNI': UNI,
    'DINOL14': DINOL14,
    'VITS8': VITS8,
    'VITS16': VITS16,
    'VITB8': VITB8,
    'VITB16': VITB16,
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet8x4_double': resnet8x4_double,
    'resnet32x4': resnet32x4,
    'resnet110x2': resnet110x2, 
    'resnet20x4': resnet20x4,
    'wrn_10_2': wrn_10_2,
    'wrn_10_1': wrn_10_1,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_28_4': wrn_28_4,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobilenet,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_0_5': ShuffleV2_0_5,
    'RegNetY_400MF': RegNetY_400MF, 
    'RegNetX_400MF': RegNetX_400MF,  
    'RegNetX_200MF': RegNetX_200MF,
    'Policy': Policy,
    'PolicyTrans': PolicyTrans
}
