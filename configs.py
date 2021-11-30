# dataset related
dataset_name = "COCO"
resize_h, resize_w = 256, 320

# dataloader related
batch_size = 128
num_workers = 12
prefetch_factor = 4

# model related
input_ch = 1 if dataset_name == "MNIST" else 3
channels = [input_ch, 32, 64, 128, 256, 512, 1024]
num_z = 1024
num_samples = 32

# training related
epochs = 100
init_lr = 1e-3
lr_decay = 0.95

# resize_h, resize_w constraints
modular = 2 ** (len(channels) - 1)
if not ((resize_h % modular == 0) and (resize_w % modular == 0)):
    assert False, "Need to change 'resize_w' or 'resize_h'"