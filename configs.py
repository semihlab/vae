# dataset related
# dataset_name = "COCO"
# resize_h, resize_w = 60, 90
dataset_name = "MNIST"
resize_h, resize_w = 28, 28

# dataloader related
batch_size = 128
num_workers = 12
prefetch_factor = 4

# model related
input_ch = 1 if dataset_name == "MNIST" else 3
channels = [input_ch, 32, 64, 128, 256, 512]
num_z = 1024
num_samples = 32

# training related
epochs = 100
init_lr = 1e-3
lr_decay = 0.95