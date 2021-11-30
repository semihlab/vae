# dataset related
dataset_name = "COCO"
resize_h, resize_w = 60, 90

# dataloader related
batch_size = 64
num_workers = 8
prefetch_factor = 4

# model related
input_ch = 1 if dataset_name == "MNIST" else 3
channels = [input_ch, 32, 64, 64, 128, 128, 256]
num_z = 1024
num_samples = 32

# training related
epochs = 30
init_lr = 1e-3
