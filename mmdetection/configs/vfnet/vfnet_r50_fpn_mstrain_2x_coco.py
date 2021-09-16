_base_ = './vfnet_r50_fpn_1x_coco.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

classes = ('opacity',)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/u/jessica63/projs/SIIM-COVID-Detection/entry/detection/coco_train_0.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/u/jessica63/projs/SIIM-COVID-Detection/entry/detection/coco_valid_0.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='',
        classes=classes,
        ann_file='/home/u/jessica63/projs/SIIM-COVID-Detection/entry/detection/coco_valid_0.json',
        pipeline=test_pipeline))
# learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

work_dir = '/work/Lung/SIIM/covid/vfnet/'
