_base_ = 'grounding_dino_swin-t_finetune_goldg_ja.py'
load_from = 'path/to/grounding_dino_swin-t_finetune_goldg_ja/model.pth'

# learning policy
max_epochs = 15

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.5),
            'language_model': dict(lr_mult=0.5),
        }))

# dataset settings
flickr30k_ja_dataset = dict(
    type='ODVGDataset',
    data_root='data/flickr30k_entities_ja/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

jcre3_dataset = dict(
    type='ODVGDataset',
    data_root='data/jcre3/',
    ann_file='mdetr_annotations_jcre3_u3s1/final_jcre3_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None
)

train_dataloader = dict(
    _delete_=True,
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset',
                 datasets=[flickr30k_ja_dataset, jcre3_dataset]))

# # NOTE: `auto_scale_lr` is for automatically scaling LR,
# # USER SHOULD NOT CHANGE ITS VALUES.
# # base_batch_size = (2 GPUs) x (12 samples per GPU)
auto_scale_lr = dict(base_batch_size=24)
