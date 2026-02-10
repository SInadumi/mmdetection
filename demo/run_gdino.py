# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo for Japanese Phrase Grounding.

This script based on demo/image_demo.py.

Example:
    Save visualizations and predictions results::

        python demo/run_gdino.py demo/demo.jpg \
        configs/mm_grounding_dino/grounding_dino_swin-t_finetune_f30k_jcre3_ja.py \
        --texts "赤い車と白い車" \
        --pred-score-thr 0.1 
"""

from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from rhoknp import KNP, Document, BasePhrase
from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.utils.util_prediction import CamelCaseDataClassJsonMixin, Rectangle, get_core_expression


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: Rectangle
    confidence: float


@dataclass(frozen=True)
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    index: int
    text: str
    bounding_boxes: list[BoundingBox]


@dataclass(frozen=True)
class GDinoPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: list[PhrasePrediction]



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, nargs="*", help='Input image file or folder path.')
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--caption-file', help='Path to Juman++ file for caption.')
    parser.add_argument(
        '--texts', help='text prompt')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--plot', action='store_true', help='Plot bboxes.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')

    return parser.parse_args()

def create_tokens_positive(caption: Document) -> list[list[list]]:
    tokens_positive: list[list[list]] = []
    char_index = 0
    for base_phrase in caption.base_phrases:
        if base_phrase.features.get("体言") is True:
            before, core, after = get_core_expression(base_phrase)
            core_start = char_index
            core_end = core_start + len(core)
            tokens_positive.append([[core_start, core_end]])
        char_index += len(base_phrase.text)
    return tokens_positive

def predict_gdino(predictions, tokens_positive, image_ids, caption):
    gdino_predictions = []
    
    base_phrase_idx_to_char_spans = []
    cursor = 0
    for base_phrase in caption.base_phrases:
        text = base_phrase.text
        start = caption.text.find(text, cursor)
        end = start + len(text)
        base_phrase_idx_to_char_spans.append((start, end))
        cursor = end

    char2base_phrase: dict[int, int] = {
        c_idx: base_phrase.index
        for base_phrase, (start, end) in zip(
            caption.base_phrases, base_phrase_idx_to_char_spans
        )
        for c_idx in range(start, end)
    }

    for image_id, pred in zip(image_ids, predictions):
        labels, scores, bboxes = pred['labels'], pred['scores'], pred['bboxes']
        assert len(labels) == len(scores) == len(bboxes)
        assert len(labels) > 0, "No predictions found."

        num_classes = max(labels) + 1
        boxes_list = [[] for _ in range(num_classes)]
        scores_list = [[] for _ in range(num_classes)]

        for idx, label in enumerate(labels):
            boxes_list[label].append(bboxes[idx])
            scores_list[label].append(scores[idx])


        phrase_index_to_bounding_boxes: dict[int, list[BoundingBox]] = defaultdict(list)
        for boxes, scores, spans in zip(boxes_list, scores_list, tokens_positive):
            bounding_boxes = [
                BoundingBox(
                    image_id=image_id,
                    rect=Rectangle(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    confidence=float(score)
                )
                for box, score in zip(boxes, scores)
            ]
            for span in spans:
                start, end = span
                base_phrase_indices = {
                    char2base_phrase[char_index]
                    for char_index in range(start, end)
                    if char_index in char2base_phrase
                }
                for base_phrase_idx in base_phrase_indices:
                    phrase_index_to_bounding_boxes[base_phrase_idx].extend(bounding_boxes)
        
        phrase_predictions: list[PhrasePrediction] = [
            PhrasePrediction(
                index=base_phrase.index,
                text=base_phrase.text,
                bounding_boxes=phrase_index_to_bounding_boxes.get(base_phrase.index, [])
            )
            for base_phrase in caption.base_phrases
        ]
        gdino_predictions.append(
            GDinoPrediction(
                doc_id=caption.doc_id,
                image_id=image_id,
                phrases=phrase_predictions
            )
        )
    return gdino_predictions

def main():
    args = parse_args()

    call_args = vars(args)
    # Do not save detection json results by predictor classes
    call_args['no_save_pred'] =True
    call_args['no_save_vis'] = True
    if call_args['plot']:
        # Do not save detection vis results
        call_args['no_save_vis'] = False
        # Color palette used for visualization
        call_args['palette'] = 'none' # or 'coco', 'voc', 'citys', 'random'
    call_args.pop('plot')

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    # Create tokens_positive
    if call_args['caption_file'] is not None:
        caption = Document.from_knp(Path(call_args['caption_file']).read_text())
    elif call_args['texts'] is not None:
        knp = KNP(options=["-dpnd-fast", "-tab"])
        caption = knp.apply_to_document(call_args['texts'])
    else:
        raise ValueError('Provide text prompts via --texts or --caption-file.')
    call_args['texts'] = caption.text
    call_args['tokens_positive'] = create_tokens_positive(caption)
    call_args.pop('caption_file')

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')

    image_ids = [Path(image_path).stem for image_path in call_args['inputs']]
    output_dir = Path(call_args['out_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Video and Webcam are currently not supported and
    #  may consume too much memory if your input folder has a lot of images.
    #  We will be optimized later.
    if len(call_args['tokens_positive']) > 0:
        inferencer = DetInferencer(**init_args)
        inferencer.model.test_cfg.chunked_size = -1
        outputs = inferencer(**call_args)

        assert len(outputs["predictions"]) == len(call_args['inputs'])
        gdino_predictions = predict_gdino(outputs["predictions"], call_args['tokens_positive'], image_ids, caption)
    else:
        gdino_predictions = [
            GDinoPrediction(
                doc_id=caption.doc_id,
                image_id=image_id,
                phrases=[
                    PhrasePrediction(
                        index=base_phrase.index,
                        text=base_phrase.text,
                        bounding_boxes=[]
                    )
                    for base_phrase in caption.base_phrases
                ]
            )
            for image_id in image_ids
        ]
    for image_id, prediction in zip(image_ids, gdino_predictions):
        output_dir.joinpath(f"{image_id}.json").write_text(prediction.to_json(indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
