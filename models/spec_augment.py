import torch
import torch.nn as nn
import random
import math
from typing import Dict, Tuple, List, Optional


class SpecAugment(nn.Module):
    """
    スペクトログラムにSpecAugmentを適用するモジュール。
    時間マスキングと周波数マスキングを行います。
    学習時(`model.train()`)にのみ適用されます。
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        p: float = 1.0,
        time_mask_ratio: Optional[float] = None,
        fixed_time_mask_size: bool = False,
    ):
        """
        Args:
            freq_mask_param (int): 周波数マスクの最大幅 (F)
            time_mask_param (int): 時間マスクの最大幅 (T)
            num_freq_masks (int): 適用する周波数マスクの数
            num_time_masks (int): 適用する時間マスクの数
            p (float): Augmentationを適用する確率
            time_mask_ratio (float | None): 時間軸で何割を隠すか。指定時は
                num_time_masks よりこちらを優先して、非重複な time mask を作る。
            fixed_time_mask_size (bool): True のとき、各 span は time_mask_param
                を基準に固定長で切る。
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        self.time_mask_ratio = (
            None if time_mask_ratio is None else float(time_mask_ratio)
        )
        self.fixed_time_mask_size = bool(fixed_time_mask_size)

        if self.time_mask_ratio is not None and not 0.0 <= self.time_mask_ratio <= 1.0:
            raise ValueError("time_mask_ratio must be between 0.0 and 1.0")

    def _apply_random_freq_masks(
        self,
        aug_spec: torch.Tensor,
        freq_mask: torch.Tensor,
        batch_index: int,
        num_mels: int,
    ) -> None:
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f == 0:
                continue
            f0 = random.randint(0, num_mels - f)
            aug_spec[batch_index, :, f0 : f0 + f, :] = 0
            freq_mask[batch_index, f0 : f0 + f] = True

    def _estimate_num_time_spans(self, num_frames: int) -> int:
        if (
            self.time_mask_ratio is None
            or self.time_mask_ratio <= 0.0
            or self.time_mask_param <= 0
            or num_frames <= 0
        ):
            return 0

        target_frames = min(
            num_frames,
            max(1, int(round(num_frames * self.time_mask_ratio))),
        )
        max_span_width = min(self.time_mask_param, num_frames)
        if self.fixed_time_mask_size:
            mean_span_width = max_span_width
        else:
            # 可変長のときは平均幅で本数を近似する。
            mean_span_width = max(1.0, (max_span_width + 1) * 0.5)

        # overlap を許容する近似版。少しだけ多めに打って実効 mask rate を近づける。
        return max(
            1,
            int(math.ceil((target_frames / mean_span_width) * 1.15)),
        )

    def _apply_time_masks(
        self,
        aug_spec: torch.Tensor,
        time_mask: torch.Tensor,
        batch_index: int,
        num_frames: int,
    ) -> None:
        if self.time_mask_ratio is not None:
            max_span_width = min(self.time_mask_param, num_frames)
            num_spans = self._estimate_num_time_spans(num_frames)
            for _ in range(num_spans):
                if self.fixed_time_mask_size:
                    width = max_span_width
                else:
                    width = random.randint(1, max_span_width)
                start = random.randint(0, num_frames - width)
                end = start + width
                aug_spec[batch_index, :, :, start:end] = 0
                time_mask[batch_index, start:end] = True
            return

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t == 0:
                continue
            t0 = random.randint(0, num_frames - t)
            aug_spec[batch_index, :, :, t0 : t0 + t] = 0
            time_mask[batch_index, t0 : t0 + t] = True

    def forward(self, spec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            spec (torch.Tensor): 入力スペクトログラム (B, C, F, T)
        Returns:
            tuple: (Augmentationが適用されたスペクトログラム, マスク情報)
        """
        device = spec.device
        batch_size, _, num_mels, num_frames = spec.shape
        freq_mask = torch.zeros(batch_size, num_mels, device=device, dtype=torch.bool)
        time_mask = torch.zeros(batch_size, num_frames, device=device, dtype=torch.bool)

        # self.trainingはnn.Moduleが持つフラグで、model.train() / model.eval()で切り替わる
        if not self.training or random.random() > self.p:
            return spec, {"freq_mask": freq_mask, "time_mask": time_mask}

        # 元のスペクトログラムをコピーして変更
        aug_spec = spec.clone()

        for i in range(batch_size):
            self._apply_random_freq_masks(
                aug_spec=aug_spec,
                freq_mask=freq_mask,
                batch_index=i,
                num_mels=num_mels,
            )
            self._apply_time_masks(
                aug_spec=aug_spec,
                time_mask=time_mask,
                batch_index=i,
                num_frames=num_frames,
            )

        return aug_spec, {"freq_mask": freq_mask, "time_mask": time_mask}


class MiniBatchMixtureMasking(nn.Module):
    """
    Mini-batch based Mixture Masking (MM).
    入力 or 隠れ表現に対して、時間/周波数の連続区間を
    同一バッチ内の他サンプルとの平均 (x + y) / 2 で置換します。
    学習時のみ適用されます。
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        p: float = 1.0,
        fallback_when_batch1: str = "zero",  # "skip" or "zero"
    ):
        """
        Args:
            freq_mask_param (int): 周波数マスクの最大幅 (F)
            time_mask_param (int): 時間マスクの最大幅 (T)
            num_freq_masks (int): 適用する周波数マスクの数
            num_time_masks (int): 適用する時間マスクの数
            p (float): Augmentationを適用する確率
            fallback_when_batch1 (str): バッチサイズが1のときの挙動
                - "skip": 何もしない
                - "zero": ゼロ詰め(ZM)にフォールバック
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p
        assert fallback_when_batch1 in ("skip", "zero")
        self.fallback_when_batch1 = fallback_when_batch1

    def forward(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): (B, C, F, T)
        Returns:
            tuple: (Augmented tensor, info dict)
                info["freq_mask"]: (B, F) bool
                info["time_mask"]: (B, T) bool
                info["partner_idx"]: (B,) long (適用時のみ有効, それ以外は-1)
        """
        device = x.device
        B, C, F, T = x.shape

        freq_mask = torch.zeros(B, F, device=device, dtype=torch.bool)
        time_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        partner_idx = torch.full((B,), -1, device=device, dtype=torch.long)

        if (not self.training) or (random.random() > self.p):
            return x, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}

        # バッチサイズ1のときの扱い
        if B < 2:
            if self.fallback_when_batch1 == "skip":
                return x, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}
            # "zero" フォールバック（ZM）
            aug = x.clone()
            for i in range(B):
                for _ in range(self.num_freq_masks):
                    f = random.randint(0, self.freq_mask_param)
                    if f > 0:
                        f0 = random.randint(0, F - f)
                        aug[i, :, f0 : f0 + f, :] = 0
                        freq_mask[i, f0 : f0 + f] = True
                for _ in range(self.num_time_masks):
                    t = random.randint(0, self.time_mask_param)
                    if t > 0:
                        t0 = random.randint(0, T - t)
                        aug[i, :, :, t0 : t0 + t] = 0
                        time_mask[i, t0 : t0 + t] = True
            return aug, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}

        # 通常: MM を適用
        aug = x.clone()
        # partner をサンプルごとに一人だけ選ぶ（マスク数に関わらず固定）
        if group_ids is not None:
            group_ids = group_ids.to(device)
            # group_id -> [indices] を作る
            groups = {}
            for i in range(B):
                g = int(group_ids[i].item())
                groups.setdefault(g, []).append(i)

            # 各グループ内でペアを選ぶ（同ステム同士のみ）
            for g, idxs in groups.items():
                n = len(idxs)
                if n == 1:
                    # グループに1件しかないときはスキップ（必要ならゼロ詰めにするなど拡張可）
                    continue
                for i in idxs:
                    if n == 2:
                        j = idxs[1] if i == idxs[0] else idxs[0]
                    else:
                        # 自分以外からランダムに選ぶ
                        while True:
                            j = random.choice(idxs)
                            if j != i:
                                break
                    partner_idx[i] = j
        else:
            # バッチ全体から相手を選ぶ
            for i in range(B):
                j = random.randrange(B - 1)
                if j >= i:
                    j += 1
                partner_idx[i] = j

        # 元の x を参照して置換（連続マスクの順序で結果が歪まないように）
        for i in range(B):
            y = x[int(partner_idx[i].item())]  # (C, F, T)
            # 周波数方向マスク
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f == 0:
                    continue
                f0 = random.randint(0, F - f)
                # (C, f, T) を平均に置換
                aug[i, :, f0 : f0 + f, :] = 0.5 * (x[i, :, f0 : f0 + f, :] + y[:, f0 : f0 + f, :])
                freq_mask[i, f0 : f0 + f] = True

            # 時間方向マスク
            for _ in range(self.num_time_masks):
                t = random.randint(0, self.time_mask_param)
                if t == 0:
                    continue
                t0 = random.randint(0, T - t)
                # (C, F, t) を平均に置換
                aug[i, :, :, t0 : t0 + t] = 0.5 * (x[i, :, :, t0 : t0 + t] + y[:, :, t0 : t0 + t])
                time_mask[i, t0 : t0 + t] = True

        return aug, {"freq_mask": freq_mask, "time_mask": time_mask, "partner_idx": partner_idx}
