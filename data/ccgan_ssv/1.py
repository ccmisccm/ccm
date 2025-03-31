import matplotlib.pyplot as plt
from TransCGAN_model_2D_ccgan import *

patch_size = 1
seq_len = 69

# cgan model trained use wassertein loss
CGAN_model_path = r'D:\Users\Administrator\Desktop\tts-cgan-main\tts-cgan-main\logs\data augmentation_2025_03_10_09_40_35\Model\checkpoint'


gen_net = Generator(seq_len=seq_len, channels=1, num_classes=11, latent_dim=100, data_embed_dim=100,
                    label_embed_dim=10 ,depth=3, num_heads=4,
                    forward_drop_rate=0.5, attn_drop_rate=0.5)

CGAN_ckp = torch.load(CGAN_model_path)
gen_net.load_state_dict(CGAN_ckp['gen_state_dict'],strict=False)

print("模型参数是否包含NaN：", any(torch.isnan(p).any() for p in gen_net.parameters()))

for name, param in gen_net.named_parameters():
    if torch.isnan(param).any():
        total_elements = param.numel()  # 总元素数量
        nan_count = torch.isnan(param).sum().item()  # 统计 NaN 的数量
        nan_ratio = nan_count / total_elements  # 计算 NaN 占比

        print(f"⚠️ 参数 {name} 包含 {nan_count} 个 NaN，占比: {nan_ratio:.2%}")


checkpoint = torch.load(CGAN_model_path, map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())


gen_state = checkpoint.get('gen_state_dict', None)
if gen_state is not None:
    for key, param in gen_state.items():
        print(f"{key}: {param.shape}")
else:
    print("Checkpoint 中没有 'gen_state_dict' 键")


model_state = gen_net.state_dict()
for key, param in model_state.items():
    print(f"Model {key}: {param.shape}")


