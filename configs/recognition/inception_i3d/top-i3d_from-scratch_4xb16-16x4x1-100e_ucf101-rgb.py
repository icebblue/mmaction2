_base_ = [
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='Recognizer3D',
    backbone=dict(type='TopI3D'),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=1024,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/ucf101/ucf101_train_split_{split}_videos.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_videos.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=4, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5),logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
