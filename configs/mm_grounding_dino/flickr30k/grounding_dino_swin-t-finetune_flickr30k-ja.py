_base_ = '../grounding_dino_swin-t_finetune_goldg_ja.py'

test_evaluator = dict(_delete_=True, type='Flickr30kMetric')
test_dataloader = dict(
    _delete_=True,
    dataset=dict(
        type='Flickr30kDataset',
        data_root='data/flickr30k_entities_ja/',
        ann_file='final_flickr_separateGT_test.json',
        data_prefix=dict(img='flickr30k_images/'),
        pipeline=_base_.test_pipeline))
