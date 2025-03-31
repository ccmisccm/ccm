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


synthetic_data = []
synthetic_labels = []

for i in range(1):
    fake_noise = torch.FloatTensor(np.random.normal(0, 1, (200,69, 90)))
    fake_label = torch.randint(0, 5, (200,))
    fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()

    # 确保输入形状符合 gen_net 要求
    print(f"fake_noise.shape: {fake_noise.shape}")
    print(f"fake_label.shape: {fake_label.shape}")

    synthetic_data.append(fake_sigs)
    synthetic_labels.append(fake_label)


synthetic_data =synthetic_data
synthetic_labels=synthetic_labels