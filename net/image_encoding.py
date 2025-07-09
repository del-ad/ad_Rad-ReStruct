from datetime import datetime
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config, create_transform
from torchvision.transforms import transforms

def timestamp():
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
class ImageEncoderEfficientNet(nn.Module):
    def __init__(self, args):
        super(ImageEncoderEfficientNet, self).__init__()

        self.args = args
        self.model = timm.create_model('tf_efficientnet_b5', pretrained=True)
        config = resolve_data_config({}, model=self.model)
        self.transforms = create_transform(**config)
        self.missing_knowledge_embedding = torch.tensor([[[ 2.9669e-02,  6.0119e-02, -2.5112e-02,  1.2975e-02,  3.7170e-02,
          -1.6165e-02,  4.9672e-02,  1.5843e-02,  7.8692e-02, -3.6440e-02,
           8.8433e-03, -1.6865e-02,  2.8844e-02, -4.8447e-02, -7.7059e-02,
           9.4805e-03, -9.6113e-03, -6.2877e-02, -2.5904e-02, -5.6922e-02,
           4.2831e-02,  7.0799e-03, -6.1605e-03,  1.7923e-02, -8.9002e-03,
           3.5839e-02, -2.7328e-02,  4.5749e-02, -3.7951e-02,  4.3477e-02,
          -1.6380e-02, -3.1858e-03,  2.7937e-02, -3.1492e-03, -5.4538e-03,
          -2.9281e-02, -2.7260e-02, -7.6457e-02, -7.1021e-03,  8.0159e-03,
          -1.0089e-02,  1.5996e-02,  6.7148e-02,  5.1516e-03,  3.1801e-02,
          -3.7696e-02, -2.4550e-02, -3.4798e-02,  7.1074e-02, -3.3497e-02,
          -2.6948e-02,  4.9287e-02,  3.9357e-02, -4.1347e-02,  8.2677e-03,
          -3.4628e-02, -2.9921e-02,  2.4188e-02, -8.6922e-03,  1.3824e-02,
           3.6691e-03,  4.2375e-02,  9.4308e-03,  1.0627e-02, -5.1713e-02,
          -3.9155e-02, -5.2533e-02, -2.4861e-02,  6.6356e-03, -3.5741e-02,
          -3.0885e-02, -4.0743e-03, -3.5157e-02, -1.0714e-02,  3.7846e-02,
           2.5153e-02, -3.5409e-02,  2.9503e-02, -1.7956e-02,  1.8888e-02,
          -1.3498e-02, -5.0194e-02,  6.9361e-03,  2.7228e-02,  5.6588e-03,
          -1.7501e-02, -4.3600e-02,  3.3361e-02, -5.4103e-02,  2.3982e-02,
          -1.1956e-02,  4.1684e-03, -1.3184e-02,  2.7170e-02,  3.2067e-02,
          -1.0623e-01,  5.1153e-02, -3.2244e-02,  2.9272e-02,  5.5884e-02,
          -2.0693e-02, -5.3499e-02, -3.4590e-02,  2.3446e-02, -3.4673e-02,
          -2.1067e-02, -3.1294e-02, -7.9230e-03, -2.2388e-03,  3.6977e-02,
           3.9103e-02, -3.0174e-02, -3.3818e-02, -4.7989e-02,  4.2760e-02,
          -5.0523e-04,  4.1016e-02,  9.3231e-03,  1.9547e-02, -6.3306e-02,
          -1.0689e-02, -5.4750e-02,  6.9551e-03, -3.2561e-02, -1.0845e-02,
          -3.1421e-02, -1.6428e-02, -3.2349e-02, -2.9069e-02,  9.2095e-03,
           5.1760e-02, -9.7557e-04, -5.9022e-04, -3.9404e-02,  1.3756e-03,
           5.5149e-03,  1.8352e-03, -4.2567e-02, -3.9484e-02, -3.0062e-03,
           4.3819e-02,  1.0559e-02, -2.4574e-02, -3.2498e-02, -9.1579e-03,
           1.9606e-02, -4.3247e-02,  1.7315e-02,  2.7389e-02, -9.9641e-03,
          -5.9764e-03, -4.0797e-03, -4.3696e-02,  5.1164e-02,  6.6715e-02,
           3.8144e-02,  9.2762e-03,  1.2339e-02, -1.6152e-02, -1.1387e-02,
           6.4506e-02,  2.1704e-02,  3.3132e-02,  2.8269e-02,  6.0430e-02,
          -1.4987e-02,  5.3808e-02,  1.8236e-03, -5.6370e-02,  1.7063e-02,
           6.1496e-03, -3.9213e-02, -2.5820e-03,  7.1148e-02,  2.8557e-02,
          -7.2259e-02, -1.4518e-04,  3.2128e-02, -3.1063e-02,  6.0677e-02,
           2.4173e-02, -2.0389e-02,  8.1892e-02,  5.6122e-02, -2.1911e-02,
           3.4344e-02,  2.9801e-02,  6.3769e-02,  6.1858e-02,  2.0184e-02,
          -7.2330e-02, -2.9019e-02,  1.3439e-02, -1.5256e-02, -1.5852e-02,
          -8.8134e-03,  4.8555e-03,  3.1473e-02, -2.0762e-03,  4.3517e-02,
           1.2584e-02,  1.5345e-03,  1.4686e-02,  2.3571e-02, -2.7673e-02,
          -2.0314e-02, -3.9804e-02, -1.0902e-02,  5.6489e-02, -1.0153e-02,
          -3.2529e-02,  6.3112e-02,  7.3950e-02,  3.3800e-02, -1.5595e-03,
          -1.9192e-02,  3.4079e-02, -3.7838e-02,  5.7854e-03,  4.0337e-04,
           2.7126e-02,  6.7723e-02,  4.5926e-02, -2.2802e-02, -1.0473e-03,
          -3.8487e-02, -2.1401e-02, -1.2116e-02,  1.5060e-02, -7.8374e-03,
          -2.0513e-02, -1.1829e-02, -3.1022e-02, -6.4077e-03,  1.2934e-02,
           3.6033e-03,  1.3409e-02, -4.8748e-03, -5.1525e-02,  2.9871e-02,
          -1.4771e-02, -2.1663e-02, -3.2677e-02, -7.6233e-02, -5.0058e-05,
          -5.4310e-02, -1.8257e-02,  4.3329e-02, -5.3151e-02, -4.7203e-02,
           2.8226e-02,  1.7948e-02,  5.7990e-03, -7.8148e-03,  2.5811e-02,
           7.1527e-02, -5.0625e-03, -2.3985e-02, -9.5200e-03,  5.9870e-03,
          -1.8198e-02, -6.4945e-02,  4.5645e-02,  5.6343e-02, -6.4075e-03,
           5.9558e-02,  1.7073e-02, -3.3875e-02, -3.8389e-02, -3.2552e-02,
          -3.0393e-02,  6.4300e-03,  8.1190e-03, -5.0736e-02,  2.3552e-02,
          -6.2450e-03,  4.3829e-02,  4.0507e-02, -5.4959e-02,  3.3773e-02,
           2.4986e-02,  1.4678e-02,  5.1510e-02,  1.6664e-02, -1.9277e-02,
          -2.8012e-02,  7.8492e-02,  4.3185e-02,  2.6758e-03, -1.5803e-02,
           3.2977e-02,  4.4630e-02,  3.2231e-02,  3.9125e-02,  1.9288e-02,
           2.9872e-02, -4.0787e-02,  5.4731e-03,  2.2710e-02,  5.7142e-02,
          -3.6869e-03,  6.2076e-03, -5.4474e-02, -9.3317e-03, -4.2247e-03,
           4.3136e-02, -4.3207e-02,  1.9942e-02, -6.3697e-02, -2.4098e-02,
          -3.3486e-02, -2.7706e-02,  3.2665e-02,  3.4804e-02, -2.3554e-02,
          -7.3436e-02, -2.9757e-02, -1.5466e-02,  2.2882e-02,  1.9675e-02,
          -3.8003e-02, -4.6917e-02, -6.0785e-02,  8.3079e-02,  2.8001e-02,
          -9.2506e-02,  3.2684e-02, -4.8887e-02, -4.4994e-02,  3.1398e-02,
           9.4325e-03,  1.1644e-02,  1.0506e-02,  5.2548e-02, -2.0345e-02,
           6.9859e-03, -7.0205e-02, -4.7934e-03, -1.8173e-02, -1.3696e-02,
          -2.8408e-02,  1.9745e-02,  2.3493e-02,  3.2479e-02,  2.8077e-03,
           5.8954e-03,  3.0508e-02,  1.3816e-02,  7.0711e-02, -3.2608e-02,
          -2.5357e-02,  4.0752e-02,  1.0176e-03,  2.9313e-02,  5.4776e-02,
           1.3180e-02, -5.9508e-02,  1.2825e-02,  1.9574e-02, -1.3197e-02,
          -2.1004e-02, -4.9882e-02, -4.5760e-03,  4.2886e-02, -6.2251e-02,
          -3.2731e-02,  4.8599e-02,  2.7346e-02,  2.0580e-02, -4.6106e-03,
          -7.0019e-02,  3.4100e-03, -4.0208e-02, -9.3445e-03,  2.0134e-03,
           8.8158e-02,  6.2274e-02, -4.5329e-02,  2.5995e-02,  1.7628e-02,
           4.0438e-03,  7.2643e-02,  7.5985e-02, -1.3048e-02, -5.8811e-02,
           4.3947e-02, -7.1773e-02,  1.2625e-02, -7.3148e-03, -3.6544e-02,
          -2.6545e-02,  1.1601e-02,  1.8480e-02,  3.8215e-03,  1.9748e-02,
           3.4152e-02, -1.4939e-02,  2.8268e-02, -3.0260e-02, -7.6495e-03,
           6.8348e-02,  2.3168e-02,  4.4217e-03,  1.6263e-02, -8.0949e-02,
           3.8653e-02,  3.2239e-02,  3.7965e-02,  4.8978e-02, -7.3137e-02,
          -3.8951e-02,  3.2380e-02,  1.2885e-02,  2.3176e-04, -4.2906e-03,
          -1.0532e-02,  3.9823e-02, -2.1200e-02, -1.4618e-02,  5.2068e-02,
          -6.8766e-02,  4.1433e-02, -4.5131e-02,  1.1998e-02,  2.0060e-02,
          -2.4108e-02, -4.2492e-02,  1.2103e-02, -1.6704e-02, -3.4516e-02,
          -1.2916e-02,  1.7342e-02,  3.8315e-02, -2.8264e-02, -7.3410e-02,
          -1.6944e-02, -6.0143e-02, -2.2207e-02, -3.0399e-02,  4.4668e-02,
          -6.5278e-03, -2.8444e-02, -2.2374e-02, -4.6733e-02, -1.3289e-02,
          -4.0788e-06,  7.9633e-03,  6.9638e-02,  3.6667e-02,  3.0172e-02,
           2.3586e-02, -3.3327e-02,  2.0884e-02,  3.3523e-02,  3.8484e-02,
          -2.8116e-02,  2.1768e-02, -1.8732e-02,  1.7807e-02, -3.7200e-02,
          -2.3821e-02, -9.4530e-03,  4.2800e-03,  4.4900e-02, -5.3556e-03,
          -6.8440e-02,  3.1485e-04,  1.5669e-02,  4.9225e-02, -1.0441e-03,
          -4.7917e-02, -4.4169e-02,  2.4306e-02,  2.9520e-04,  1.1613e-02,
          -3.7752e-02,  5.4865e-02, -1.8858e-02,  4.3476e-02,  5.7561e-02,
           6.1475e-02,  4.7650e-03,  6.6733e-02,  3.2651e-02, -2.0392e-02,
           1.0100e-02,  7.6072e-02, -7.8451e-02,  4.2851e-02, -2.0291e-02,
          -2.7177e-02, -4.8095e-03,  3.6416e-03,  1.6594e-02, -4.4802e-02,
          -6.1687e-03, -1.1068e-02,  5.1676e-02, -4.3477e-03, -3.9871e-02,
           5.2286e-02, -1.0339e-02, -3.4308e-02, -1.0262e-02,  8.6397e-03,
          -4.5065e-02,  8.0973e-03, -4.1055e-03,  2.5779e-02,  3.6899e-02,
          -2.4197e-02, -5.8654e-02,  8.5665e-02,  3.7391e-03, -2.7227e-02,
           2.5397e-03,  1.8721e-02,  7.3946e-02,  6.4267e-02, -2.4393e-02,
          -4.9167e-03,  3.1750e-02, -2.7031e-03,  1.2968e-02, -5.2228e-02,
          -3.3566e-02,  6.6195e-02, -1.9900e-02,  2.2754e-02,  5.1048e-02,
           2.6649e-02,  2.5427e-02,  5.2028e-03,  9.7495e-03, -7.3259e-02,
          -4.6176e-02, -1.3302e-02, -3.7683e-02, -3.2823e-02,  3.4041e-02,
           1.1838e-02,  5.6483e-03,  3.4905e-02, -1.3168e-02,  9.1205e-03,
           1.1127e-02,  5.3055e-02, -2.4856e-03, -1.5405e-02, -5.3950e-02,
          -3.2472e-02,  6.7028e-02, -2.0616e-02,  2.8538e-03, -4.9519e-03,
          -3.2897e-02,  7.1753e-02, -3.3810e-02,  3.6435e-03, -3.5530e-02,
          -7.8728e-03, -5.0257e-02,  8.2671e-02, -5.8616e-02,  7.1690e-02,
          -3.4478e-03, -3.4806e-02,  4.5581e-02,  1.9257e-02,  4.5025e-02,
           2.0549e-02,  2.3099e-02, -2.2967e-02, -6.6774e-03, -4.0645e-02,
          -3.7270e-03, -1.0914e-02,  2.9412e-02, -2.2725e-02, -1.7072e-02,
          -9.3725e-03,  1.3047e-02,  5.0040e-02, -1.3954e-02, -2.2752e-02,
          -5.3096e-03, -6.7337e-02, -3.4459e-02,  2.0747e-02, -1.0094e-02,
          -4.5311e-02, -3.3037e-02, -2.4434e-02,  6.0652e-04, -7.6677e-03,
          -2.3077e-02,  6.7045e-02, -6.2990e-04, -5.3641e-02, -3.9134e-02,
           2.8619e-02, -2.9418e-02, -5.1317e-02,  2.9650e-02,  2.8975e-02,
          -3.0362e-02,  7.3118e-03, -4.6363e-03, -9.3448e-03,  3.9381e-02,
           6.1424e-02, -8.7402e-02, -6.8233e-02, -3.1352e-02, -1.6088e-02,
           2.9652e-02,  4.6844e-02, -7.1275e-02, -2.9319e-02,  1.9243e-03,
          -6.4359e-02, -5.0629e-02, -9.1760e-03,  4.0406e-03, -1.4547e-02,
           5.6510e-02, -1.8483e-02, -1.0784e-02,  2.0285e-02,  7.7146e-03,
          -1.5399e-02, -4.2171e-02, -2.1199e-02,  7.1944e-02,  4.4867e-02,
          -6.7441e-02,  1.2392e-02, -2.4278e-02, -7.6761e-02, -1.7166e-02,
          -6.1458e-02,  9.8877e-03, -4.7967e-02, -1.4160e-02,  1.5348e-02,
          -4.8479e-02, -5.9549e-02,  3.6965e-02, -1.8745e-02,  2.8953e-02,
          -1.6789e-02, -4.6658e-02, -6.3460e-02,  8.4475e-03,  2.5892e-02,
          -2.4717e-02, -3.9033e-02, -3.0492e-02, -1.5596e-02, -1.6432e-02,
          -6.2069e-02, -8.1662e-02, -3.0501e-02,  1.2223e-02, -3.0374e-03,
           1.9149e-03,  2.7935e-03, -1.0070e-02, -4.5117e-02,  8.5111e-02,
           8.0039e-02,  3.6676e-02,  4.7646e-02, -8.5912e-03, -4.2785e-02,
           8.1952e-02, -3.7995e-03, -4.4631e-02,  3.9114e-02,  2.2078e-02,
          -2.4807e-03,  2.5061e-02,  2.5385e-03, -4.3261e-03, -4.4362e-02,
           4.5567e-02, -3.9050e-03, -1.1448e-02,  3.7744e-02,  2.9206e-02,
          -1.5660e-02,  3.4416e-02, -1.9481e-02, -6.3467e-02, -3.2164e-02,
          -2.4507e-02, -4.6913e-02,  2.0046e-02, -5.4143e-02,  7.0361e-03,
          -3.9957e-02,  6.8750e-02,  4.5871e-02, -2.3101e-02, -1.8761e-02,
          -2.8881e-03,  3.2908e-02,  2.0326e-02,  3.6822e-02, -4.1539e-03,
           9.1609e-03,  6.2151e-03, -6.6799e-02,  1.8646e-02, -1.3184e-02,
           3.6556e-02,  4.6001e-02,  2.2897e-04, -1.7413e-03, -4.6291e-05,
          -5.3284e-03, -2.6355e-03,  1.6794e-04,  3.9607e-02,  1.3014e-02,
          -7.8891e-02, -4.2266e-02,  3.4043e-02,  1.9977e-02,  1.5405e-02,
          -4.4330e-02,  5.2248e-02,  5.1651e-03, -3.0041e-02,  4.9966e-03,
          -4.4950e-03, -7.1887e-02, -5.7171e-03, -1.4464e-02, -3.8425e-02,
           3.8965e-02, -3.2620e-02,  4.3147e-02,  4.7408e-02,  1.0649e-02,
          -1.7819e-02, -3.2081e-02, -3.6022e-02,  1.6267e-02,  3.3724e-03,
          -6.1091e-03,  6.0316e-03,  3.9381e-03,  2.5815e-02, -1.1996e-02,
           6.0470e-03,  2.4862e-02, -4.2253e-02, -2.5355e-03,  1.0772e-02,
          -1.0591e-03,  1.5921e-02, -2.7625e-02, -3.5394e-02,  2.6525e-02,
          -1.0573e-02, -2.0254e-03, -2.5659e-02]]])
        self.missing_knowledge_embedding = F.normalize(self.missing_knowledge_embedding, dim=-1)
        self.zero_embedding = torch.zeros_like(self.missing_knowledge_embedding)
        self.one_embedding = torch.ones_like(self.missing_knowledge_embedding)
        #self.missing_knowledge_embedding = self.missing_knowledge_embedding.to(device=self.model.device)
        if 'radrestruct' in args.data_dir:
            self.transforms.transforms[0] = transforms.Resize((488, 488))

        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.relu = nn.ReLU()
        ### Original
        self.rescale_conv = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        ### New
        # self.rescale_conv = nn.Sequential(
        #     nn.Conv2d(2048, args.hidden_size, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True), 
        #     nn.BatchNorm2d(args.hidden_size)  # Helps with AMP stability
        # )
        self.rescale_pool = nn.AvgPool2d(kernel_size=2, stride=1)

        # transforms
        self.img_tfm = transforms.Compose(self.transforms.transforms[:-1])
        self.norm_tfm = self.transforms.transforms[-1]
        self.resize_size = self.img_tfm.transforms[1].size  # size of CenterCrop
        
    # def forward(self, img, mode='train'):
    #     x = self.model(img)
    #     x = self.rescale_conv(x)
    #     x = self.rescale_pool(x)

    #     # Token embedding (flatten spatial dimensions into sequence)
    #     image_tokens = x.flatten(2).permute(0, 2, 1)  # (B, h'*w', hidden_size)

    #     return image_tokens

    ### New
    def forward(self, img, mode='train'):
        ### Traces    
        # try:
        #     self.eval()
        #     with torch.no_grad():
        #         img1_b4_model = img[0:1]
        #         img2_b4_model = img[1:2]
        #         print(f"Before model: {((img1_b4_model - img2_b4_model).abs().max())}")
                
        #         x = self.model(img)
        #         img1_b4_rescale = x[0:1]
        #         img2_b4_rescale = x[1:2]
        #         print(f"After model: {((img1_b4_rescale - img2_b4_rescale).abs().max())}")
        #         #x = self.rescale_conv(x)
        #         x = self.rescale_conv(x)
                
        #         img1_after_rescale = x[0:1]
        #         img2_after_rescale = x[1:2]
        #         print(f"After rescale: {((img1_after_rescale - img2_after_rescale).abs().max())}")
                
        #         x = self.rescale_pool(x)
                
        #         img1_after_pool = x[0:1]
        #         img2_after_pool = x[1:2]
        #         print(f"After pool: {((img1_after_pool - img2_after_pool).abs().max())}")
        #         # Clamp and sanitize values before global mean (important for AMP)
        #         x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                
        #         img1_after_nan = x[0:1]
        #         img2_after_nan = x[1:2]
        #         print(f"After nan: {((img1_after_nan - img2_after_nan).abs().max())}")
        #         x = torch.clamp(x, min=-1e4, max=1e4)
                
        #         img1_after_clamp = x[0:1]
        #         img2_after_clamp = x[1:2]
        #         print(f"After clamp: {((img1_after_clamp - img2_after_clamp).abs().max())}")
        #         # Global embedding (average over spatial dimensions) - Global everage pooling
        #         global_embedding = x.amax(dim=[2, 3])  # (B, hidden_size)
                
        #         img1_after_ge = x[0:1]
        #         img2_after_ge = x[1:2]
        #         print(f"After global embedding: {((img1_after_ge - img2_after_ge).abs().max())}")
        #         global_embedding = global_embedding.unsqueeze(1) # (B, 1, hidden_size)

        #         # Token embedding (flatten spatial dimensions into sequence)
        #         image_tokens = x.flatten(2).permute(0, 2, 1)  # (B, h'*w', hidden_size)
        # finally:
        #     if was_training:
        #         self.train()
                
        x = self.model(img)
        x = self.rescale_conv(x)
        x = self.rescale_pool(x)
        # Clamp and sanitize values before global mean (important for AMP)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-1e4, max=1e4)

        # Global embedding (average over spatial dimensions) - Global everage pooling
        # global_embedding = x.mean(dim=[2, 3])  # (B, hidden_size)
        # global_embedding = global_embedding.unsqueeze(1) # (B, 1, hidden_size)
        
        # Global embedding (max over spatial dimensions) - Global everage pooling
        if self.args.use_pretrained:
            pass
        else:
            global_embedding = x.amax(dim=[2, 3])  # (B, hidden_size)
            global_embedding = global_embedding.unsqueeze(1) # (B, 1, hidden_size)

        # Token embedding (flatten spatial dimensions into sequence)
        image_tokens = x.flatten(2).permute(0, 2, 1)  # (B, h'*w', hidden_size)
                
        if not torch.isfinite(image_tokens).all():
            print(f'{timestamp()}Image encoder produced nan/inf in image_tokens')

        # if not torch.isfinite(global_embedding).all():
        #     print(f'{timestamp()}Image encoder produced nan/inf in global_embedding')

        return image_tokens if self.args.use_pretrained else image_tokens ,global_embedding
    
    @torch.no_grad()
    def get_global_embeddings(self, img):
        was_training = self.training
            
        try:
            self.eval()
            with torch.autocast('cuda', torch.float16):
                x = self.model(img)
                #x = self.rescale_conv(x)
                x = self.rescale_conv(x)
                x = self.rescale_pool(x)

            # Clamp and sanitize values before global mean (important for AMP)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, min=-1e4, max=1e4)

            # Global embedding (average over spatial dimensions) - Global everage pooling
            global_embedding = x.amax(dim=[2, 3])  # (B, hidden_size)
            global_embedding = global_embedding.unsqueeze(1) # (B, 1, hidden_size)

            # # Token embedding (flatten spatial dimensions into sequence)
            # image_tokens = x.flatten(2).permute(0, 2, 1)  # (B, h'*w', hidden_size)
            
        finally:
            if was_training:
                self.train()
             

        return global_embedding
    
    # def encode_and_max_pool(self, img_batch):
    #     """
    #     img_batch: Tensor of shape (N, 3, H, W), where N is the number of images
    #     encoder: your ImageEncoderEfficientNet model

    #     Returns:
    #         Tensor of shape (1, 1, hidden_dim)
    #     """
    #     # Run the encoder on the batch of N images
    #     _, global_embeddings = self.forward(img_batch)  # (N, 1, hidden_dim)

    #     # Remove singleton dim for pooling
    #     global_embeddings = global_embeddings.squeeze(1)  # (N, hidden_dim)

    #     # Max-pool across N images
    #     pooled, _ = torch.max(global_embeddings, dim=0)  # (hidden_dim,)

    #     # Reshape to (1, 1, hidden_dim)
    #     return pooled.unsqueeze(0).unsqueeze(0)
    
# class FrozenImageEncoderEfficientNet(nn.Module):
    
#     def __init__(self, args):
#         super(FrozenImageEncoderEfficientNet, self).__init__()

#         self.args = args
#         self.model = timm.create_model('tf_efficientnet_b5', pretrained=True)
#         config = resolve_data_config({}, model=self.model)
#         self.transforms = create_transform(**config)
#         self.missing_knowledge_embedding = F.normalize(torch.randn(1, 1, args.hidden_size), dim=-1)
#         if 'radrestruct' in args.data_dir:
#             self.transforms.transforms[0] = transforms.Resize((488, 488))

#         self.model = nn.Sequential(*list(self.model.children())[:-2])

#         self.relu = nn.ReLU()
#         self.rescale_conv = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         self.rescale_pool = nn.AvgPool2d(kernel_size=2, stride=1)

#         # transforms
#         self.img_tfm = transforms.Compose(self.transforms.transforms[:-1])
#         self.norm_tfm = self.transforms.transforms[-1]
#         self.resize_size = self.img_tfm.transforms[1].size  # size of CenterCrop
        
#         for param in self.parameters():
#             param.requires_grad = False

#         self.eval()

#     def forward(self, img):
#         with torch.no_grad():
#             x = self.model(img)
#             x = self.rescale_conv(x)
#             x = self.rescale_pool(x)

#             # Global embedding (average over spatial dimensions) - Global everage pooling
#             global_embeddings = x.mean(dim=[2, 3])  # (B, hidden_size)
#             global_embeddings = global_embeddings.unsqueeze(1) # (B, 1, hidden_size)

#         return global_embeddings
    
#     def encode_and_max_pool(self, img_batch):
#         """
#         img_batch: Tensor of shape (N, 3, H, W), where N is the number of images
#         encoder: your ImageEncoderEfficientNet model

#         Returns:
#             Tensor of shape (1, 1, hidden_dim)
#         """
#         # Run the encoder on the batch of N images
#         _, global_embeddings = self.forward(img_batch)  # (N, 1, hidden_dim)

#         # Remove singleton dim for pooling
#         global_embeddings = global_embeddings.squeeze(1)  # (N, hidden_dim)

#         # Max-pool across N images
#         pooled, _ = torch.max(global_embeddings, dim=0)  # (hidden_dim,)

#         # Reshape to (1, 1, hidden_dim)
#         return pooled.unsqueeze(0).unsqueeze(0)


def adapt_position_encoding(model, patch_size=32, after=384, suffix='visual.positional_embedding'):
    keys = [k for k in model if k.endswith(suffix)]
    assert len(keys) == 1
    key = keys[0]
    origin_pos_embed = model[key]
    origin_dim2 = False
    if len(origin_pos_embed.shape) == 2:
        origin_dim2 = True
        origin_pos_embed = origin_pos_embed.unsqueeze(0)
    grid_before = int(np.sqrt(origin_pos_embed.shape[1] - 1))
    before = int(grid_before * patch_size)
    assert (before % patch_size) == 0
    grid_after = after // patch_size
    assert (after % patch_size) == 0
    embed_dim = origin_pos_embed.shape[-1]

    pos_embed = origin_pos_embed[0, 1:, :].reshape((grid_before, grid_before, embed_dim))
    new_size = (grid_after, grid_after)
    pos_embed = torch.nn.functional.interpolate(pos_embed.permute((2, 0, 1)).unsqueeze(0), size=new_size,
                                                mode='bicubic')
    pos_embed = pos_embed.squeeze(0).permute((1, 2, 0)).reshape((-1, embed_dim))
    pos_embed = torch.cat((origin_pos_embed[0, 0:1, :], pos_embed), dim=0).unsqueeze(0)
    assert pos_embed.shape == (1, grid_after * grid_after + 1, embed_dim)
    if origin_dim2:
        assert pos_embed.shape[0] == 1
        pos_embed = pos_embed.squeeze(0)
    model[key] = pos_embed
    return model
