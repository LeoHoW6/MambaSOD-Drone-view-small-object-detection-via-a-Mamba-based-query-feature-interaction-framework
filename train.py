import os
import math
import time
import random
import argparse
import datetime
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from scipy.optimize import linear_sum_assignment

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
vim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vim')
if vim_dir not in sys.path:
    sys.path.insert(0, vim_dir)
from model import MambaSOD

from vim_pretrained_loader import load_vim_pretrained


# ================================================================
#  Box utilities
# ================================================================

def box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:,:,0] * wh[:,:,1]
    giou = iou - (enclosing - union) / (enclosing + 1e-7)
    return giou


# ================================================================
#  Hungarian Matcher
# ================================================================

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=2.0, cost_giou=5.0,
                 num_classes=10, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Nq = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].sigmoid()
        out_bbox = outputs['pred_boxes']

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        indices = []
        for b in range(B):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            prob = out_prob[b]
            neg_cost = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
            pos_cost = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
            cost_cls = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

            cost_l1 = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox))

            C = (self.cost_class * cost_cls +
                 self.cost_bbox * cost_l1 +
                 self.cost_giou * cost_giou).cpu()

            C[C.isnan() | C.isinf()] = 1e4

            r, c_ = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long),
                            torch.as_tensor(c_, dtype=torch.long)))
        return indices


# ================================================================
#  Set Criterion
# ================================================================

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses,
                 focal_alpha=0.25, focal_gamma=1.5, class_freq=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if class_freq is not None:
            freq = torch.tensor(class_freq, dtype=torch.float32)
            inv_sqrt = 1.0 / freq.sqrt()
            alpha_per_class = inv_sqrt * focal_alpha / inv_sqrt.mean()
            alpha_per_class = alpha_per_class.clamp(min=0.05, max=0.8)
            self.register_buffer('alpha_per_class', alpha_per_class)
        else:
            self.alpha_per_class = None

    def loss_labels(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        B, Nq, C = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])

        target_classes = torch.full(
            (B, Nq), self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            (B, Nq, self.num_classes + 1),
            dtype=pred_logits.dtype, device=pred_logits.device
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        gamma = self.focal_gamma
        prob = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_onehot, reduction='none'
        )
        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        focal = ce_loss * ((1 - p_t) ** gamma)

        if self.alpha_per_class is not None:
            pos_alpha = self.alpha_per_class.view(1, 1, -1)
            neg_alpha = 1.0 - self.focal_alpha
            alpha_t = pos_alpha * target_onehot + neg_alpha * (1 - target_onehot)
        else:
            alpha = self.focal_alpha
            alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)

        focal = alpha_t * focal

        loss_ce = focal.mean(1).sum() / max(num_boxes, 1)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)

        if len(src_boxes) == 0:
            dev = outputs['pred_boxes'].device
            return {'loss_bbox': torch.tensor(0., device=dev),
                    'loss_giou': torch.tensor(0., device=dev)}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(num_boxes, 1)

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / max(num_boxes, 1)

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        src_idx = torch.cat([s for (s, _) in indices])
        return batch_idx, src_idx

    def _compute_losses_single(self, outputs, targets, num_boxes):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss_type == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def _group_losses(self, outputs, targets, num_boxes, num_groups):
        if num_groups == 1:
            return self._compute_losses_single(outputs, targets, num_boxes)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        B, total_Nq, C = pred_logits.shape
        Nq = total_Nq // num_groups

        pred_logits_g = pred_logits.reshape(B, num_groups, Nq, C)
        pred_boxes_g = pred_boxes.reshape(B, num_groups, Nq, 4)

        accumulated = {}
        for g in range(num_groups):
            group_out = {
                'pred_logits': pred_logits_g[:, g],
                'pred_boxes': pred_boxes_g[:, g],
            }
            group_losses = self._compute_losses_single(group_out, targets, num_boxes)
            for k, v in group_losses.items():
                if k not in accumulated:
                    accumulated[k] = v
                else:
                    accumulated[k] = accumulated[k] + v

        for k in accumulated:
            accumulated[k] = accumulated[k] / num_groups

        return accumulated

    def forward(self, outputs, targets):
        num_groups = outputs.get('num_groups', 1)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs['pred_logits'].device
        ).clamp(min=1).item()

        losses = self._group_losses(outputs, targets, num_boxes, num_groups)

        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                aux_losses = self._group_losses(aux_out, targets, num_boxes, num_groups)
                losses.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})

        return losses


# ================================================================
#  VisDrone Dataset
# ================================================================

class VisDroneDataset(Dataset):
    CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    IGNORED = [0, 11]

    def __init__(self, root_dir, split='train', img_size=640,
                 transform=None, use_mosaic=True, mosaic_prob=0.5):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.use_mosaic = use_mosaic and (split == 'train')
        self.mosaic_prob = mosaic_prob

        base = os.path.join(root_dir, f'VisDrone2019-DET-{split}')
        for img_name in ['image', 'images']:
            p = os.path.join(base, img_name)
            if os.path.exists(p):
                self.img_dir = p
                break
        else:
            raise FileNotFoundError(f"Image directory not found: {base}")

        for anno_name in ['annotation', 'annotations']:
            p = os.path.join(base, anno_name)
            if os.path.exists(p):
                self.anno_dir = p
                break
        else:
            raise FileNotFoundError(f"Annotation directory not found: {base}")

        self.img_files = sorted([f for f in os.listdir(self.img_dir)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        assert len(self.img_files) > 0, f"Empty directory: {self.img_dir}"

        print(f"[{split}] Loading annotations for {len(self.img_files)} images...")
        self.annos = []
        for img_f in self.img_files:
            anno_f = os.path.splitext(img_f)[0] + '.txt'
            self.annos.append(self._parse(os.path.join(self.anno_dir, anno_f)))

    def _parse(self, path):
        boxes, labels = [], []
        if not os.path.exists(path):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        with open(path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat = int(parts[5])
                if cat in self.IGNORED or w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(cat - 1)
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def _load_img(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_files[idx]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resize(self, img, boxes):
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= self.img_size / ow
            boxes[:, [1, 3]] *= self.img_size / oh
        return img, boxes

    def _mosaic(self, idx):
        ids = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        s2 = self.img_size * 2
        out = np.zeros((s2, s2, 3), np.uint8)
        cx = s2 // 2 + random.randint(-s2 // 8, s2 // 8)
        cy = s2 // 2 + random.randint(-s2 // 8, s2 // 8)
        all_b, all_l = [], []

        for i, id_ in enumerate(ids):
            img = self._load_img(id_)
            bx, lb = self.annos[id_]
            bx, lb = bx.copy(), lb.copy()
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a = max(cx-w, 0), max(cy-h, 0)
                x2a, y2a = cx, cy
                x1b, y1b = w-(x2a-x1a), h-(y2a-y1a)
                x2b, y2b = w, h
            elif i == 1:
                x1a, y1a = cx, max(cy-h, 0)
                x2a, y2a = min(cx+w, s2), cy
                x1b, y1b = 0, h-(y2a-y1a)
                x2b, y2b = x2a-x1a, h
            elif i == 2:
                x1a, y1a = max(cx-w, 0), cy
                x2a, y2a = cx, min(cy+h, s2)
                x1b, y1b = w-(x2a-x1a), 0
                x2b, y2b = w, y2a-y1a
            else:
                x1a, y1a = cx, cy
                x2a, y2a = min(cx+w, s2), min(cy+h, s2)
                x1b, y1b = 0, 0
                x2b, y2b = x2a-x1a, y2a-y1a

            out[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(bx) > 0:
                bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], x1b, x2b)
                bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], y1b, y2b)
                keep = ((bx[:, 2]-bx[:, 0]) > 2) & ((bx[:, 3]-bx[:, 1]) > 2)
                bx, lb = bx[keep], lb[keep]
                if len(bx) > 0:
                    bx[:, [0, 2]] += x1a - x1b
                    bx[:, [1, 3]] += y1a - y1b
                    all_b.append(bx)
                    all_l.append(lb)

        out = cv2.resize(out, (self.img_size, self.img_size))
        sc = self.img_size / s2
        if all_b:
            all_b = np.concatenate(all_b) * sc
            all_l = np.concatenate(all_l)
        else:
            all_b = np.zeros((0, 4), np.float32)
            all_l = np.zeros((0,), np.int64)
        return out, all_b, all_l

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob:
            img, boxes, labels = self._mosaic(idx)
        else:
            img = self._load_img(idx)
            boxes, labels = self.annos[idx]
            boxes, labels = boxes.copy(), labels.copy()
            img, boxes = self._resize(img, boxes)

        if self.transform:
            t = self.transform(image=img,
                               bboxes=boxes.tolist() if len(boxes) > 0 else [],
                               class_labels=labels.tolist() if len(labels) > 0 else [])
            img = t['image']
            if t['bboxes']:
                boxes = np.array(t['bboxes'], np.float32)
                labels = np.array(t['class_labels'], np.int64)
            else:
                boxes = np.zeros((0, 4), np.float32)
                labels = np.zeros((0,), np.int64)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        if len(boxes) > 0:
            boxes[:, [0, 2]] /= self.img_size
            boxes[:, [1, 3]] /= self.img_size
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)
            valid = (boxes[:, 2] > 1e-4) & (boxes[:, 3] > 1e-4)
            boxes, labels = boxes[valid], labels[valid]

        return {'image': img, 'boxes': boxes, 'labels': labels, 'image_id': idx}

    @staticmethod
    def collate_fn(batch):
        return {
            'images': torch.stack([x['image'] for x in batch]),
            'boxes': [x['boxes'] for x in batch],
            'labels': [x['labels'] for x in batch],
            'image_ids': [x['image_id'] for x in batch],
        }


def get_train_transforms(img_size=640):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussNoise(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'],
                                 min_visibility=0.2, min_area=4))

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def build_dataloaders(args):
    train_ds = VisDroneDataset(
        args.data_root, 'train', args.img_size,
        transform=get_train_transforms(args.img_size),
        use_mosaic=args.use_mosaic, mosaic_prob=args.mosaic_prob,
    )
    val_ds = VisDroneDataset(
        args.data_root, 'val', args.img_size,
        transform=get_val_transforms(),
        use_mosaic=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def prepare_targets(batch, device):
    return [{'labels': l.to(device), 'boxes': b.to(device)}
            for b, l in zip(batch['boxes'], batch['labels'])]


# ================================================================
#  mAP Evaluation (COCO 101-point interpolation)
# ================================================================

def _compute_ap_101(recall, precision):
    """
    COCO-style 101-point interpolated AP.

    AP = (1/101) * sum_{r in {0, 0.01, 0.02, ..., 1.0}} p_interp(r)
    where p_interp(r) = max_{r' >= r} p(r')
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for t in recall_thresholds:
        inds = np.where(mrec >= t)[0]
        if len(inds) > 0:
            ap += mpre[inds[0]]
    ap /= 101.0
    return ap


@torch.no_grad()
def compute_map(model, data_loader, device, num_classes=10,
                iou_thresh=0.5, score_thresh=0.05, img_size=640):
    """mAP@iou_thresh using 101-point interpolation, split by object size."""
    model.eval()

    buckets = ['all', 'small', 'medium', 'large']
    all_preds = {bk: {c: [] for c in range(num_classes)} for bk in buckets}
    all_ngt = {bk: {c: 0 for c in range(num_classes)} for bk in buckets}

    def get_bucket(area_px):
        if area_px < 1024: return 'small'
        elif area_px < 9216: return 'medium'
        else: return 'large'

    for batch in data_loader:
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        outputs = model(images)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        B = pred_logits.shape[0]
        fg_probs = pred_logits.sigmoid()

        for b in range(B):
            max_scores, max_cls = fg_probs[b].max(dim=-1)
            keep = max_scores > score_thresh
            p_scores = max_scores[keep]
            p_cls = max_cls[keep]
            p_boxes = pred_boxes[b][keep]
            order = p_scores.argsort(descending=True)
            p_scores = p_scores[order]
            p_cls = p_cls[order]
            p_boxes = p_boxes[order]

            gt_labels = targets[b]['labels']
            gt_boxes = targets[b]['boxes']
            if len(gt_boxes) > 0:
                gt_area_px = (gt_boxes[:, 2] * gt_boxes[:, 3]) * (img_size ** 2)
                gt_bucket_list = [get_bucket(a.item()) for a in gt_area_px]
            else:
                gt_bucket_list = []

            for i in range(len(gt_labels)):
                c = gt_labels[i].item()
                all_ngt['all'][c] += 1
                all_ngt[gt_bucket_list[i]][c] += 1

            if len(gt_labels) == 0:
                for i in range(len(p_scores)):
                    c = p_cls[i].item()
                    all_preds['all'][c].append((p_scores[i].item(), False))
                continue
            if len(p_boxes) == 0:
                continue

            p_xyxy = box_cxcywh_to_xyxy(p_boxes)
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
            iou_mat, _ = box_iou(p_xyxy, gt_xyxy)

            gt_matched = {bk: torch.zeros(len(gt_labels), dtype=torch.bool, device=device)
                          for bk in buckets}

            for i in range(len(p_scores)):
                c = p_cls[i].item()
                score = p_scores[i].item()
                for bk in buckets:
                    if bk == 'all':
                        bk_mask = (gt_labels == c) & (~gt_matched['all'])
                    else:
                        bk_area_mask = torch.tensor(
                            [gt_bucket_list[j] == bk for j in range(len(gt_labels))],
                            device=device, dtype=torch.bool)
                        bk_mask = (gt_labels == c) & bk_area_mask & (~gt_matched[bk])

                    if not bk_mask.any():
                        if bk == 'all':
                            all_preds[bk][c].append((score, False))
                        continue

                    ious = iou_mat[i] * bk_mask.float()
                    best_gt = ious.argmax().item()
                    if ious[best_gt] >= iou_thresh:
                        all_preds[bk][c].append((score, True))
                        gt_matched[bk][best_gt] = True
                    elif bk == 'all':
                        all_preds[bk][c].append((score, False))

    class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                   'truck', 'tricycle', 'awning-tri', 'bus', 'motor']
    results = {}
    for bk in buckets:
        aps = []
        header = f"mAP@{iou_thresh}" if bk == 'all' else f"AP_{bk}@{iou_thresh}"
        print(f"\n  === {header} (101-pt) ===")
        for c in range(num_classes):
            preds = all_preds[bk][c]
            ngt = all_ngt[bk][c]
            if ngt == 0:
                continue
            preds.sort(key=lambda x: -x[0])
            tp = np.array([int(p[1]) for p in preds])
            fp = 1 - tp
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / ngt
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            ap = _compute_ap_101(recall, precision)
            aps.append(ap)
            if bk == 'all':
                print(f"      {class_names[c]:>12s}: AP={ap:.4f} (nGT={ngt}, nDet={len(preds)})")
        mAP = np.mean(aps) if aps else 0
        print(f"         {bk:>6s} mAP: {mAP:.4f}")
        results[bk] = mAP

    return results['all']


# ================================================================
#  Training Engine
# ================================================================

def train_one_epoch(model, criterion, loader, optimizer, scaler,
                    device, epoch, args):
    model.train()
    criterion.train()

    stats = {'loss': 0, 'ce': 0, 'bbox': 0, 'giou': 0, 'n': 0}
    t0 = time.time()
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        if sum(len(t['labels']) for t in targets) == 0:
            continue

        with autocast(enabled=args.amp):
            outputs = model(images)
            ld = criterion(outputs, targets)

            w = criterion.weight_dict
            loss = sum(ld[k] * w[k] for k in ld if k in w)
            for k in ld:
                if 'aux' in k:
                    bk = k.rsplit('_aux', 1)[0]
                    if bk in w:
                        loss = loss + ld[k] * w[bk]

            loss = loss / args.grad_accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            print(f"  NaN/Inf at batch {i+1}, skipping")
            continue

        loss = torch.clamp(loss, max=500.0)

        scaler.scale(loss).backward()

        if (i + 1) % args.grad_accum_steps == 0:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        stats['loss'] += loss.item() * args.grad_accum_steps
        stats['ce'] += ld.get('loss_ce', torch.tensor(0)).item()
        stats['bbox'] += ld.get('loss_bbox', torch.tensor(0)).item()
        stats['giou'] += ld.get('loss_giou', torch.tensor(0)).item()
        stats['n'] += 1

        if (i+1) % args.print_freq == 0:
            n = stats['n']
            el = time.time() - t0
            eta = el / (i+1) * (len(loader) - i - 1)
            print(f"  [{i+1}/{len(loader)}] "
                  f"loss={stats['loss']/n:.4f} "
                  f"ce={stats['ce']/n:.4f} "
                  f"bbox={stats['bbox']/n:.4f} "
                  f"giou={stats['giou']/n:.4f} "
                  f"ETA={datetime.timedelta(seconds=int(eta))}")

    n = max(stats['n'], 1)
    return {k: stats[k]/n for k in ['loss', 'ce', 'bbox', 'giou']}


@torch.no_grad()
def evaluate(model, criterion, loader, device, args):
    model.eval()
    criterion.eval()
    stats = {'loss': 0, 'ce': 0, 'bbox': 0, 'giou': 0, 'n': 0}

    for batch in loader:
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        if sum(len(t['labels']) for t in targets) == 0:
            continue

        with autocast(enabled=args.amp):
            outputs = model(images)
            ld = criterion(outputs, targets)

        w = criterion.weight_dict
        loss = sum(ld[k] * w[k] for k in ld if k in w)

        if not torch.isnan(loss):
            stats['loss'] += loss.item()
            stats['ce'] += ld.get('loss_ce', torch.tensor(0)).item()
            stats['bbox'] += ld.get('loss_bbox', torch.tensor(0)).item()
            stats['giou'] += ld.get('loss_giou', torch.tensor(0)).item()
            stats['n'] += 1

    n = max(stats['n'], 1)
    r = {k: stats[k]/n for k in ['loss', 'ce', 'bbox', 'giou']}
    print(f"  [Val] loss={r['loss']:.4f} ce={r['ce']:.4f} "
          f"bbox={r['bbox']:.4f} giou={r['giou']:.4f}")
    return r


# ================================================================
#  LR Scheduler / Checkpoint
# ================================================================

def build_lr_scheduler(optimizer, args):
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_ckpt(model, optimizer, scheduler, scaler, epoch, best, args, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best,
        'args': vars(args),
    }, path)


def load_ckpt(model, optimizer, scheduler, scaler, path, device,
              model_only=False):
    ck = torch.load(path, map_location=device)
    state = ck.get('model_state_dict', ck.get('model'))
    model_dict = model.state_dict()

    skip_keys = []
    for k in state:
        if k in model_dict:
            if state[k].shape != model_dict[k].shape:
                skip_keys.append(k)

    filtered = {k: v for k, v in state.items() if k not in skip_keys}
    model.load_state_dict(filtered, strict=False)

    if model_only:
        print(f"  Loaded model weights from {path}")
        return 0, float('inf')
    else:
        opt_key = 'optimizer_state_dict' if 'optimizer_state_dict' in ck else 'optimizer'
        sch_key = 'scheduler_state_dict' if 'scheduler_state_dict' in ck else 'scheduler'
        sca_key = 'scaler_state_dict' if 'scaler_state_dict' in ck else 'scaler'
        if opt_key in ck:
            try: optimizer.load_state_dict(ck[opt_key])
            except: pass
        if sch_key in ck:
            try: scheduler.load_state_dict(ck[sch_key])
            except: pass
        if sca_key in ck:
            try: scaler.load_state_dict(ck[sca_key])
            except: pass
        print(f"  Resumed from {path}")
        return ck.get('epoch', 0), ck.get('best_loss', float('inf'))


# ================================================================
#  Main
# ================================================================

def main(args):
    print("=" * 70)
    print("MambaSOD Training")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} images, "
          f"Val: {len(val_loader.dataset)} images")

    model = MambaSOD(
        img_size=args.img_size, patch_size=16,
        vim_depth=24, vim_embed_dim=192, d_state=16,
        out_channels=256, bifpn_repeats=3,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.d_ffn,
        num_classes=args.num_classes,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        num_groups=args.num_groups,
        use_ca=args.use_ca,
        use_sa=args.use_sa,
        use_fpn=args.use_fpn,
    )

    if args.vim_pretrained:
        if not os.path.exists(args.vim_pretrained):
            raise FileNotFoundError(f"Pretrained weights not found: {args.vim_pretrained}")
        load_vim_pretrained(model.encoder.backbone, args.vim_pretrained)

    model = model.to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        num_classes=args.num_classes,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    weight_dict = {
        'loss_ce': args.w_ce,
        'loss_bbox': args.w_bbox,
        'loss_giou': args.w_giou,
    }
    visdrone_class_freq = [8844, 5125, 1287, 14064, 1975, 750, 1045, 532, 251, 4886]

    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes'],
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        class_freq=visdrone_class_freq if args.class_balanced else None,
    ).to(device)

    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if 'encoder.backbone' in n and p.requires_grad],
         'lr': args.lr * args.backbone_lr_mult, 'name': 'backbone'},
        {'params': [p for n, p in model.named_parameters()
                    if 'encoder.backbone' not in n and p.requires_grad],
         'lr': args.lr, 'name': 'rest'},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        ep, bl = load_ckpt(model, optimizer, scheduler, scaler,
                           args.resume, device, model_only=args.model_only)
        if not args.model_only:
            start_epoch = ep + 1
            best_loss = bl

        new_backbone_lr = args.lr * args.backbone_lr_mult
        new_rest_lr = args.lr

        for pg in optimizer.param_groups:
            if pg.get('name') == 'backbone':
                pg['lr'] = new_backbone_lr
                pg['initial_lr'] = new_backbone_lr
            else:
                pg['lr'] = new_rest_lr
                pg['initial_lr'] = new_rest_lr

        new_base_lrs = []
        for pg in optimizer.param_groups:
            if pg.get('name') == 'backbone':
                new_base_lrs.append(new_backbone_lr)
            else:
                new_base_lrs.append(new_rest_lr)
        scheduler.base_lrs = new_base_lrs

        best_loss = float('inf')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTraining: ep {start_epoch}->{args.epochs}, "
          f"bs={args.batch_size}x{args.grad_accum_steps}"
          f"={args.batch_size*args.grad_accum_steps}")

    log = []
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.freeze_backbone_epochs:
            for n, p in model.named_parameters():
                if 'encoder.backbone' in n:
                    p.requires_grad = False
                if 'reference_point' in n:
                    p.requires_grad = False
        else:
            for n, p in model.named_parameters():
                if 'encoder.backbone' in n:
                    p.requires_grad = True
                if 'reference_point' in n:
                    p.requires_grad = True

        t0 = time.time()
        lr = optimizer.param_groups[1]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={lr:.2e}")

        ts = train_one_epoch(model, criterion, train_loader, optimizer,
                             scaler, device, epoch, args)
        scheduler.step()

        vs = evaluate(model, criterion, val_loader, device, args)

        elapsed = time.time() - t0
        print(f"  Time: {datetime.timedelta(seconds=int(elapsed))}")

        log.append({'epoch': epoch+1, 'train': ts, 'val': vs, 'lr': lr})
        with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        if vs['loss'] < best_loss:
            best_loss = vs['loss']
            save_ckpt(model, optimizer, scheduler, scaler, epoch,
                      best_loss, args, os.path.join(args.output_dir, 'best.pth'))
            print(f"  Best: {best_loss:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            save_ckpt(model, optimizer, scheduler, scaler, epoch,
                      best_loss, args,
                      os.path.join(args.output_dir, f'ep{epoch+1}.pth'))

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            print(f"  Computing mAP@0.5 (101-pt)...")
            mAP = compute_map(model, val_loader, device,
                              num_classes=args.num_classes)

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    save_ckpt(model, optimizer, scheduler, scaler, args.epochs-1,
              best_loss, args, os.path.join(args.output_dir, 'last.pth'))
    print(f"\nDone! Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser('MambaSOD')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--use_mosaic', action='store_true', default=True)
    p.add_argument('--no_mosaic', dest='use_mosaic', action='store_false')
    p.add_argument('--mosaic_prob', type=float, default=0.5)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--num_queries', type=int, default=300)
    p.add_argument('--num_decoder_layers', type=int, default=6)
    p.add_argument('--d_ffn', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--drop_path_rate', type=float, default=0.1)
    p.add_argument('--cost_class', type=float, default=2.0)
    p.add_argument('--cost_bbox', type=float, default=2.0)
    p.add_argument('--cost_giou', type=float, default=5.0)
    p.add_argument('--w_ce', type=float, default=2.0)
    p.add_argument('--w_bbox', type=float, default=2.0)
    p.add_argument('--w_giou', type=float, default=5.0)
    p.add_argument('--focal_alpha', type=float, default=0.25)
    p.add_argument('--focal_gamma', type=float, default=1.5)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum_steps', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone_lr_mult', type=float, default=0.1)
    p.add_argument('--freeze_backbone_epochs', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--num_groups', type=int, default=6)
    p.add_argument('--use_ca', action='store_true', default=False)
    p.add_argument('--use_sa', action='store_true', default=False)
    p.add_argument('--use_fpn', action='store_true', default=False)
    p.add_argument('--vim_pretrained', type=str, default='')
    p.add_argument('--output_dir', type=str, default='./output/mambasod')
    p.add_argument('--print_freq', type=int, default=50)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--eval_freq', type=int, default=10)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--model_only', action='store_true', default=False)
    p.add_argument('--class_balanced', action='store_true', default=False)

    main(p.parse_args())