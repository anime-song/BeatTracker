import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Union

try:
    from .spec_augment import SpecAugment
    from .cqt import RecursiveCQT
    from .transformer import RMSNorm, Transformer
except ImportError:
    from models.spec_augment import SpecAugment
    from models.cqt import RecursiveCQT
    from models.transformer import RMSNorm, Transformer


def checkpoint_bypass(func, *args, **kwargs):
    """チェックポイントを使用しない場合のバイパス関数"""
    return func(*args)


class OctaveSharedAggregate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        f_total: int,
        hidden_size: int,
        bins_per_octave: int = 36,
        octave_emb_dim: int = 32,
        conv_kernel_size: int = 3,
        dropout: float = 0.0,
        gate_hidden_factor: float = 0.5,
        use_film: bool = True,
        return_weights: bool = False,
    ):
        super().__init__()
        assert f_total % bins_per_octave == 0, "F_total must be divisible by bins_per_octave"
        self.in_channels = in_channels
        self.f_total = f_total
        self.hidden = hidden_size
        self.bins_per_octave = bins_per_octave
        self.num_octaves = f_total // bins_per_octave
        self.use_film = use_film
        self.return_weights = return_weights

        pad = conv_kernel_size // 2

        def _choose_gn_groups(channels: int, max_groups: int = 8) -> int:
            # GroupNormのgroupsは channels を割り切る必要あり
            for g in reversed(range(1, max_groups + 1)):
                if channels % g == 0:
                    return g
            return 1

        gn_groups = _choose_gn_groups(hidden_size, max_groups=8)

        # Shared conv applied per octave: (B*O, C, T, bins) -> (B*O, hidden, T, bins)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=conv_kernel_size, padding=pad),
            nn.GroupNorm(gn_groups, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Octave index embedding -> FiLM params (gamma, beta) per octave
        #    gamma/beta are channel-wise (hidden) and broadcast over (T, bins).
        self.oct_emb = nn.Embedding(self.num_octaves, octave_emb_dim)

        if use_film:
            self.film = nn.Sequential(
                nn.Linear(octave_emb_dim, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
            )
        else:
            # additive bias (weaker but simpler)
            self.oct_add = nn.Linear(octave_emb_dim, hidden_size)

        # Data-dependent octave gate: softmax over octaves
        gate_h = max(8, int(hidden_size * gate_hidden_factor))
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, gate_h),
            nn.GELU(),
            nn.Linear(gate_h, 1),
        )

    def forward(self, x: torch.Tensor):
        B, C, T, F = x.shape

        # (B, C, T, (O*bins)) -> (B, O, C, T, bins)
        x = einops.rearrange(x, "b c t (o bins) -> b o c t bins", o=self.num_octaves, bins=self.bins_per_octave)

        # merge (B,O) for shared conv
        octave_features = []
        for i in range(self.num_octaves):
            octave_features.append(self.shared(x[:, i]))
        x = torch.stack(octave_features, dim=1)

        # octave index embedding injection
        octave_ids = torch.arange(self.num_octaves, device=x.device)  # (O,)
        emb = self.oct_emb(octave_ids)  # (O, E)
        emb = emb.unsqueeze(0).expand(B, -1, -1)  # (B, O, E)

        if self.use_film:
            film = self.film(emb)  # (B, O, 2H)
            gamma, beta = film.chunk(2, dim=-1)  # (B, O, H), (B, O, H)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            x = x * (1.0 + torch.tanh(gamma)) + beta
        else:
            add = self.oct_add(emb).unsqueeze(-1).unsqueeze(-1)  # (B, O, H, 1, 1)
            x = x + add

        # gate weights from per-octave summary
        # summary: (B, O, H)
        summary = x.mean(dim=(-1, -2))  # average over (T,bins) -> (B, O, H)
        w = self.gate(summary)  # (B, O, 1)
        w = torch.softmax(w, dim=1)  # softmax over octaves
        w5 = w.unsqueeze(-1).unsqueeze(-1)  # (B, O, 1, 1, 1)

        # weighted sum over octave dim -> (B, H, T, bins)
        y = (x * w5).sum(dim=1)

        if self.return_weights:
            # return weights as (B, O)
            return y, w.squeeze(-1)
        return y


class AudioFeatureExtractor(nn.Module):
    """
    音声波形から特徴量(CQTスペクトログラム)を抽出するクラス。
    SpecAugmentや標準化も行う。
    """

    def __init__(
        self,
        sampling_rate: int,
        n_fft: int,
        hop_length: int,
        num_audio_channels: int = 12,
        num_stems: int = 6,
        bins_per_octave: int = 12 * 3,
        n_bins: int = 12 * 3 * 7,
        spec_augment_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_audio_channels = num_audio_channels
        self.num_stems = num_stems

        if self.num_stems <= 0:
            raise ValueError("num_stems must be a positive integer")
        if self.num_audio_channels % self.num_stems != 0:
            raise ValueError(
                f"num_audio_channels ({self.num_audio_channels}) must be divisible by num_stems ({self.num_stems})"
            )
        self.channels_per_stem = self.num_audio_channels // self.num_stems

        self.bins_per_octave = int(bins_per_octave)
        self.n_bins = int(n_bins)
        if self.bins_per_octave <= 0 or self.bins_per_octave % 12 != 0:
            raise ValueError("bins_per_octave must be a positive multiple of 12")
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        self.cqt = RecursiveCQT(
            sr=sampling_rate,
            hop_length=hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            filter_scale=0.4375,
        )

        self.spec_augment = SpecAugment(**spec_augment_params) if spec_augment_params else None

    def forward(self, waveform: torch.Tensor) -> "BackboneContext":
        # center=True の STFT を想定しているため、ラベルと一致するようにクロップする
        crop_length = (waveform.shape[-1] - self.n_fft) // self.hop_length + 1

        waveform_per_stem = einops.rearrange(
            waveform, "b (s c) t -> b s c t", s=self.num_stems, c=self.channels_per_stem
        )

        if self.training:
            batch_size = waveform.shape[0]
            stem_specs: List[torch.Tensor] = []
            for stem_idx in range(self.num_stems):
                stem_waveform = waveform_per_stem[:, stem_idx]  # [B, C_per_stem, T]
                stem_flat = einops.rearrange(stem_waveform, "b c t -> (b c) t")
                stem_cqt = self.cqt(stem_flat.float(), return_complex=False)
                stem_cqt = einops.rearrange(
                    stem_cqt, "(b c) f t -> b c f t", b=batch_size, c=self.channels_per_stem
                ).contiguous()

                if self.spec_augment is not None:
                    stem_cqt_aug, _ = self.spec_augment(stem_cqt)
                    stem_specs.append(stem_cqt_aug)
                    del stem_cqt, stem_cqt_aug
                else:
                    stem_specs.append(stem_cqt)
                    del stem_cqt
            spec = torch.cat(stem_specs, dim=1)
        else:
            waveform_flat = einops.rearrange(waveform, "b c t -> (b c) t")
            spec = self.cqt(waveform_flat.float(), return_complex=False)
            spec = einops.rearrange(spec, "(b c) f t -> b c f t", c=self.num_audio_channels)

        # 標準化(バッチ単位の全体平均/分散)
        mean = spec.mean(dim=(2, 3), keepdim=True)
        std = spec.std(dim=(2, 3), keepdim=True) + 1e-8
        spec = (spec - mean) / std
        spec = spec.to(waveform.dtype)

        spec = einops.rearrange(spec, "b c f t -> b c t f").contiguous()

        original_time_steps = spec.shape[-2]

        return BackboneContext(
            spec=spec,
            crop_length=crop_length,
            original_time_steps=original_time_steps,
        )


@dataclass
class BackboneContext:
    spec: torch.Tensor
    crop_length: int
    original_time_steps: int


@dataclass
class BeatTranscriptionOutput:
    logits: torch.Tensor
    beat_logits: torch.Tensor
    downbeat_logits: torch.Tensor
    frame_features: Optional[torch.Tensor] = None
    context_features: Optional[torch.Tensor] = None
    intermediate_features: Optional[List[torch.Tensor]] = None


class Backbone(nn.Module):
    def __init__(
        self,
        feature_extractor: AudioFeatureExtractor,
        hidden_size: int,
        output_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.hidden_size = hidden_size
        self.output_dim = output_dim if output_dim is not None else hidden_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        self.num_audio_channels = feature_extractor.num_audio_channels
        self.n_bins = feature_extractor.n_bins

        self.oct_frontend = OctaveSharedAggregate(
            in_channels=self.num_audio_channels,  # 12
            f_total=self.n_bins,  # 252
            hidden_size=hidden_size,
            bins_per_octave=feature_extractor.bins_per_octave,
            octave_emb_dim=32,
            dropout=dropout,
            use_film=True,
        )

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1, stride=(2, 1)),
            nn.GroupNorm(4, hidden_size * 2),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 2,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(
                hidden_size * 4,
                hidden_size * 4,
                kernel_size=3,
                padding=1,
                stride=(2, 1),
            ),
            nn.GroupNorm(4, hidden_size * 4),
            nn.GELU(),
            nn.Conv2d(hidden_size * 4, hidden_size * 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_size * 4),
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            time_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // 8,
                num_layers=1,
                num_heads=8,
                ffn_hidden_size_factor=2,
                dropout=dropout,
            )
            band_roformer = Transformer(
                input_dim=hidden_size * 4,
                head_dim=hidden_size * 4 // 8,
                num_layers=1,
                num_heads=8,
                ffn_hidden_size_factor=2,
                dropout=dropout,
            )
            self.layers.append(nn.ModuleList([time_roformer, band_roformer]))
        self.final_norm = RMSNorm(hidden_size * 4)

        freq_downsample_factor = 1
        input_freq_bins = feature_extractor.bins_per_octave
        output_freq_bins = input_freq_bins // freq_downsample_factor

        final_channels = hidden_size * 4
        flattened_dim = output_freq_bins * final_channels

        self.to_time_features = nn.Conv1d(flattened_dim, self.output_dim, kernel_size=1)

        self.up_time = nn.ConvTranspose1d(
            self.output_dim,
            self.output_dim,
            kernel_size=8,
            stride=8,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        context: Optional[BackboneContext] = None,
        return_intermediate: bool = False,
        return_context: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if context is None:
            context = self.feature_extractor(waveform)

        use_checkpoint = self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        checkpoint_fn = torch.utils.checkpoint.checkpoint if use_checkpoint else checkpoint_bypass

        x = self.oct_frontend(context.spec)  # (B, hidden_size, T, 36)
        x = self.conv1(x)
        # ダウンサンプリング
        x = self.down_conv(x)
        x = einops.rearrange(x, "b c t f -> b t f c")  # (B, downT, F, C)

        intermediate_features = []

        for time_roformer, band_roformer in self.layers:
            B, T, freq, C = x.shape
            # 周波数軸Transformer
            x = x.reshape(B * T, freq, C)  # [B*T, F, C]
            x = checkpoint_fn(band_roformer, x, use_reentrant=False)
            x = x.reshape(B, T, freq, C)

            # 時間軸Transformer
            x = einops.rearrange(x, "b t f c -> (b f) t c")  # [B*F, T, C]
            x = checkpoint_fn(time_roformer, x, use_reentrant=False)
            x = einops.rearrange(x, "(b f) t c -> b t f c", f=freq)  # [B, T, F, C]

            if return_intermediate:
                intermediate_features.append(x)

        x = self.final_norm(x)  # [B, T, F, D]

        x = einops.rearrange(x, "b t f d -> b (f d) t")  # [B, F*D, downT]
        x = self.to_time_features(x)  # [B, output_dim, downT]

        context_features = einops.rearrange(x, "b d t -> b t d")  # [B, downT, output_dim]

        x = self.up_time(x)  # [B, output_dim, downT*8]

        target_T = context.crop_length
        if x.shape[-1] < target_T:
            x = F.pad(x, (0, target_T - x.shape[-1]))
        else:
            x = x[..., :target_T]

        x = einops.rearrange(x, "b d t -> b t d")  # [B, T, output_dim]

        if return_intermediate:
            return x, intermediate_features

        if return_context:
            return x, context_features

        return x


class BeatDownbeatHead(nn.Module):
    """
    Backbone が出力したフレーム特徴から beat / downbeat のロジットを生成するヘッド。
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.beat_head = nn.Linear(input_dim, 1)
        self.downbeat_head = nn.Linear(input_dim, 1)

    def forward(self, frame_features: torch.Tensor) -> BeatTranscriptionOutput:
        x = self.norm(frame_features)
        x = self.dropout(x)

        beat_logits = self.beat_head(x).squeeze(-1)
        downbeat_logits = self.downbeat_head(x).squeeze(-1)
        logits = torch.stack([beat_logits, downbeat_logits], dim=-1)

        return BeatTranscriptionOutput(
            logits=logits,
            beat_logits=beat_logits,
            downbeat_logits=downbeat_logits,
        )


class BeatTranscriptionModel(nn.Module):
    """
    分離済み stem 波形から beat / downbeat を予測するモデル。
    Backbone は時系列特徴の抽出を担当し、このクラスで最終ロジットへ写像する。
    """

    def __init__(self, backbone: Backbone, head_dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.head = BeatDownbeatHead(backbone.output_dim, dropout=head_dropout)

    def forward(
        self,
        waveform: torch.Tensor,
        context: Optional[BackboneContext] = None,
        return_features: bool = False,
        return_context: bool = False,
        return_intermediate: bool = False,
    ) -> BeatTranscriptionOutput:
        if return_context and return_intermediate:
            raise ValueError("return_context and return_intermediate cannot both be True")

        context_features: Optional[torch.Tensor] = None
        intermediate_features: Optional[List[torch.Tensor]] = None

        if return_intermediate:
            frame_features, intermediate_features = self.backbone(
                waveform,
                context=context,
                return_intermediate=True,
            )
        elif return_context:
            frame_features, context_features = self.backbone(
                waveform,
                context=context,
                return_context=True,
            )
        else:
            frame_features = self.backbone(waveform, context=context)

        output = self.head(frame_features)
        if return_features:
            output.frame_features = frame_features
        output.context_features = context_features
        output.intermediate_features = intermediate_features
        return output
