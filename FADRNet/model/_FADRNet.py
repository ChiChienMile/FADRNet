import torch
from torch import nn
import torch.nn.functional as F

from model.Backbone.SENet import SENet3D
from model.Backbone.SENet import SENetBottleneck
from model.Backbone.FreqBackbone_tiny import FreqBackbone, FreqDilatedBottleneck

# from Backbone.SENet import SENet3D
# from Backbone.SENet import SENetBottleneck
# from Backbone.FreqBackbone_tiny import FreqBackbone, FreqDilatedBottleneck

class BackBone3D(nn.Module):
    """
    3D SENet backbone wrapper that exposes intermediate layers.
    """
    def __init__(self, in_channels=2, stride_list=[2, 2, 2, 2], channel_list=[64, 128, 256, 512, 512]):
        super(BackBone3D, self).__init__()
        # Build a 3D SENet (ResNet-50-like: [3, 4, 6, 3]).
        net = SENet3D(
            SENetBottleneck, [3, 4, 6, 3], num_classes=2,
            in_channels=in_channels,
            channel_list=channel_list, stride_list=stride_list
        )
        # Flatten children to pick specific stage blocks.
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])  # stem
        self.layer1 = nn.Sequential(*net[3:5]) # stage1
        self.layer2 = net[5]                   # stage2
        self.layer3 = net[6]                   # stage3
        self.layer4 = net[7]                   # stage4

    def forward(self, x):
        # Return the final feature map (before global pooling).
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

class PreIncrNet(nn.Module):
    """
    Pre-incremental network (teacher or base student before adaptation).
    Provides a backbone + global pooling + linear classifier for the base task.
    """
    def __init__(self, basic_task, in_channels=1, n_classes=2):
        super(PreIncrNet, self).__init__()
        self.name = 'PreIncrNet_' + basic_task
        self.backbone = BackBone3D(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.CLSmodel = nn.Linear(1024, n_classes)

    def flatten(self, features):
        # Global average pooling + flatten to [B, C].
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features

    def forward(self, x, label, loss_ce):
        # Standard supervised forward for base task.
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        features = self.flatten(layer4)
        logits = self.CLSmodel(features)
        loss = loss_ce(logits, label)
        return loss, logits

    def predictcls(self, x):
        # Inference: logits for base task.
        with torch.no_grad():
            layer0 = self.backbone.layer0(x)
            layer1 = self.backbone.layer1(layer0)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            features = self.flatten(layer4)
            logits = self.CLSmodel(features)
            return logits

    def get_features(self, x):
        # Inference: pooled features for base task (not used by FADRNet).
        with torch.no_grad():
            layer0 = self.backbone.layer0(x)
            layer1 = self.backbone.layer1(layer0)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            features = self.flatten(layer4)
            return features

@torch.no_grad()
def sinkhorn_knopp(a, b, C, reg=0.1, max_iter=200, eps=1e-8):
    """
    Entropy-regularized Sinkhorn-Knopp in the primal (non-log) domain with safeguards.

    Args:
        a: [C_new] left marginal (sums to 1)
        b: [C_old] right marginal (sums to 1)
        C: [C_new, C_old] cost matrix (smaller is better)
        reg: entropy regularization strength (>0). Larger => smoother transport.
        max_iter: number of Sinkhorn iterations.
        eps: small constant for numerical stability.

    Returns:
        T: [C_new, C_old] transport matrix.
    """
    # Gibbs kernel.
    K = torch.exp(-C / reg).clamp_min(eps)  # [C_new, C_old]
    # Initialize scaling vectors u, v.
    u = torch.full_like(a, 1.0 / (a.numel()))
    v = torch.full_like(b, 1.0 / (b.numel()))
    # Iterative matrix scaling.
    for _ in range(max_iter):
        Kv = K @ v + eps
        u = a / Kv
        KTu = K.t() @ u + eps
        v = b / KTu
    T = torch.diag(u) @ K @ torch.diag(v)
    T = T / (T.sum() + eps)  # Normalize to prevent drift.
    return T


@torch.no_grad()
def optimal_transport_align_classifier(W_new, W_old, reg=0.1,
                                       a=None, b=None,
                                       squared=True, max_iter=200, eps=1e-8):
    """
    Optimal-transport alignment for classifier weights: \tilde{W} = Pi @ W_old.

    Args:
        W_new: [C_new, d] current (student) classifier weights.
        W_old: [C_old, d] previous (teacher) classifier weights.
        reg: entropy regularization (typ. 0.05~0.5).
        a: [C_new] left marginal (defaults to uniform).
        b: [C_old] right marginal (defaults to uniform).
        squared: use squared L2 distance if True (recommended).
        max_iter: Sinkhorn iterations.
        eps: numerical epsilon.

    Returns:
        Pi: [C_new, C_old] transport matrix.
        tilde_W: [C_new, d] OT-aligned old weights.
    """
    device = W_new.device
    C_new, d1 = W_new.shape
    C_old, d2 = W_old.shape
    assert d1 == d2, "W_new and W_old must have the same feature dim"

    # Cost matrix Q_{ij} = || w_i^{new} - w_j^{old} || (L2 or L2^2)
    Cmat = torch.cdist(W_new, W_old, p=2)
    if squared:
        Cmat = Cmat ** 2

    # Uniform marginals by default.
    if a is None:
        a = torch.full((C_new,), 1.0 / C_new, device=device)
    if b is None:
        b = torch.full((C_old,), 1.0 / C_old, device=device)

    # Solve OT via Sinkhorn.
    Pi = sinkhorn_knopp(a, b, C=Cmat, reg=reg, max_iter=max_iter, eps=eps)  # [C_new, C_old]

    # Aligned classifier weights.
    tilde_W = Pi @ W_old  # [C_new, d]
    return Pi, tilde_W


# Dual-Retention Mechanism (DRM): output-level KD + classifier-structure alignment KD.
class DRM(nn.Module):
    def __init__(self, n_classes=2):
        super(DRM, self).__init__()
        self.name = 'DRM'
        self.temperature = 1  # expose as hyper-parameter in practice (e.g., 2~4).
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Aligned old-classifier head for structure-preserving distillation.
        self.CLSmodel_aligned = nn.Linear(1024, n_classes)

    def kl_divergence(self, p, q):
        """
        KL divergence D(p || q) for probability tensors along last dim.
        p: target distribution
        q: student distribution
        """
        return torch.sum(p * torch.log(p / q), dim=-1).sum()

    def forward(self, x, features, student_model, teacher_model):
        # Output-level KD: teacher soft targets vs student base head on current inputs x.
        with torch.no_grad():
            logits_old = teacher_model.predictcls(x)
            p_orig = F.softmax(logits_old / self.temperature, dim=-1)
            # Classifier-structure OT alignment: align teacher weights to student's class geometry.
            W_new = student_model.CLSmodel.weight.data  # [C_new, d]
            W_old = teacher_model.CLSmodel.weight.data  # [C_old, d]
            _, tilde_W = optimal_transport_align_classifier(W_new, W_old, reg=0.1, squared=True)
            self.CLSmodel_aligned.weight.copy_(tilde_W)

        # Student predictions from base (old) head.
        logits_old = student_model.CLSmodel(features)
        # Predictions from the OT-aligned old classifier.
        logits_aligned = self.CLSmodel_aligned(features)

        p_align = F.softmax(logits_aligned / self.temperature, dim=-1)
        p_inc = F.softmax(logits_old / self.temperature, dim=-1)
        # KD: teacher vs student (output-level)
        loss_KD = self.kl_divergence(p_orig, p_inc)
        # AlignKD: student vs aligned-old (student || aligned)
        loss_alignKD = self.kl_divergence(p_inc, p_align)
        return loss_KD + loss_alignKD

    # Duplicate definition kept to match original structure; functionally identical.
    def kl_divergence(self, p, q, eps=1e-8):
        # Clamp to avoid log(0).
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        return torch.sum(p * torch.log(p / q), dim=-1).sum()


class BackBoneAdapter(nn.Module):
    """
    Frequency-conditioned adapter that injects a one-hot band indicator into channels
    and reuses late backbone blocks to produce frequency-aware features.
    """
    def __init__(self, diameters_len, in_channels, down_channels, condition=True):
        super(BackBoneAdapter, self).__init__()
        self.diameters_len = diameters_len  # one-hot length
        self.condition = condition
        # Adjust mid channels to accept concatenated condition.
        if self.condition:
            self.FreqBackbone = FreqBackbone(FreqDilatedBottleneck,
                                             in_channels=in_channels + diameters_len, down_channels=down_channels)
        else:
            self.FreqBackbone = FreqBackbone(FreqDilatedBottleneck,
                                             in_channels=in_channels, down_channels=down_channels)

    def forward(self, x, cond_idx=None):
        if self.condition:
            # x: [B, C, D, H, W]; cond_idx: int/list/tensor of band indices
            B, C, D, H, W = x.shape
            if isinstance(cond_idx, int):
                cond_idx = torch.full((B,), cond_idx, dtype=torch.long, device=x.device)
            elif isinstance(cond_idx, list):
                cond_idx = torch.tensor(cond_idx, dtype=torch.long, device=x.device)
            else:
                cond_idx = cond_idx.to(dtype=torch.long, device=x.device)
            # One-hot band indicator broadcast to spatial dims.
            one_hot = F.one_hot(cond_idx, num_classes=self.diameters_len).float()  # (B, num_freq)
            one_hot = one_hot.view(B, self.diameters_len, 1, 1, 1).expand(-1, -1, D, H, W)
            # Concatenate along channels and pass through late backbone blocks.
            x = torch.cat([x, one_hot], dim=1)  # (B, C + num_freq, D, H, W)
        x = self.FreqBackbone(x)
        return x

class FFA(nn.Module):
    """
    Frequency-Conditioned Feature Adapter (FFA):
    for each frequency band, process the features with a conditional adapter,
    then apply learnable per-(band,channel) gates and stack across bands.
    """
    def __init__(self, diameters_len, num_channels, down_channels):
        super().__init__()
        self.diameters_len = diameters_len
        # Learnable attention weights for [band, channel].
        self.att_weight = nn.Parameter(torch.ones(diameters_len, down_channels), requires_grad=True)
        # Conditional adapter marks which band the feature belongs to.
        self.Adapter = BackBoneAdapter(diameters_len=diameters_len, in_channels=num_channels, down_channels=down_channels)

    def forward(self, x):
        # x: [N, B, C, D, H, W] — N frequency bands stacked first.
        x_list = []
        for i in range(self.diameters_len):
            # Adapter consumes one band at a time: [B, C, D, H, W]
            x_list.append(self.Adapter(x[i, :], i))
        # Stack back to [B, N, C, D, H, W]
        x_list = torch.stack(x_list, dim=1)
        B, N, C, D, H, W = x_list.size()
        # Softmax over bands for each channel to ensure non-negativity and normalization.
        weight = F.softmax(self.att_weight, dim=0)      # [N, C]
        weight = weight.view(1, N, C, 1, 1, 1)          # broadcast to [B, N, C, 1, 1, 1]
        x_list = x_list + x_list * weight               # gated features
        # Permute to [N, B, C, D, H, W] for downstream FSTE interface.
        x_list = x_list.permute(1, 0, 2, 3, 4, 5)
        return x_list.contiguous()


class FSTE(nn.Module):
    """
    Frequency-Aware State Transition Estimator (FSTE):
    models smooth evolution across frequency bands via a transition network and
    an observation decoder, with uncertainty-aware noise injection and a gating fusion.
    """
    def __init__(self, num_channels, diameters_len):
        super().__init__()
        channels = num_channels
        # Transition function g_trans (conditional on band index via adapter).
        self.state_transition = BackBoneAdapter(diameters_len=diameters_len, in_channels=channels, down_channels=num_channels)
        # Process noise scale sigma_eps (one scalar per sample), squashed to (0,1).
        self.uncertainty_eps = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Observation function g_obs (decoder).
        self.observation_decoder = BackBoneAdapter(diameters_len=diameters_len, in_channels=channels, down_channels=num_channels)
        # Observation noise scale sigma_delta (0,1).
        self.uncertainty_delta = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Gating network alpha_k to fuse predicted state and observation.
        self.gate_net = nn.Sequential(
            BackBoneAdapter(diameters_len=None, in_channels=channels * 3, down_channels=channels, condition=False),
            nn.Sigmoid()
        )

    def forward(self, z_list, train_flag=True):
        # z_list: list/tuple of length N with tensors [B, C, D, H, W] per band.
        if train_flag:
            return self.predict_train(z_list)
        else:
            return self.predict_val(z_list)

    def predict_train(self, z_list):
        n = len(z_list)
        S_hat_list = [z_list[0]]  # initialize latent state with first band
        obs_losses = []
        state_losses = []

        for k in range(1, n):
            S_prev = S_hat_list[-1]  # [B, C, D, H, W]
            z_k = z_list[k]          # observation at band k

            # Predict latent state at band k from previous state.
            z_pred = self.state_transition(S_prev, k)
            sigma_eps = self.uncertainty_eps(S_prev)
            eps_k = torch.randn_like(z_pred)
            z_pred_noisy = z_pred + sigma_eps * eps_k

            # L_state: encourage predicted state to follow observed routed feature.
            state_losses.append(F.mse_loss(z_pred_noisy, z_k))

            # Decode observation from noisy predicted state.
            s_pred = self.observation_decoder(z_pred, k)
            sigma_delta = self.uncertainty_delta(z_pred)
            delta_k = torch.randn_like(s_pred)
            s_pred_noisy = s_pred + sigma_delta * delta_k

            # L_obs: anchor to observations.
            obs_losses.append(F.mse_loss(s_pred_noisy, z_k))

            # Adaptive fusion between S_pred and z_k via alpha_k.
            gate_input = torch.cat([z_pred, s_pred, z_k], dim=1)
            alpha_k = self.gate_net(gate_input)
            S_k = alpha_k * z_pred + (1 - alpha_k) * s_pred
            S_hat_list.append(S_k)

        obs_loss = torch.stack(obs_losses).mean()
        state_loss = torch.stack(state_losses).mean()
        S_hat_list = torch.stack(S_hat_list, dim=1)  # [B, N, C, D, H, W]
        return S_hat_list, obs_loss + state_loss

    def predict_val(self, z_list):
        n = len(z_list)
        S_hat_list = [z_list[0]]  # initialize latent state with first band
        for k in range(1, n):
            S_prev = S_hat_list[-1]  # [B, C, D, H, W]
            z_k = z_list[k]          # observation at band k
            # Predict latent state at band k from previous state.
            z_pred = self.state_transition(S_prev, k)
            # Decode observation from noisy predicted state.
            s_pred = self.observation_decoder(z_pred, k)
            # Adaptive fusion between S_pred and z_k via alpha_k.
            gate_input = torch.cat([z_pred, s_pred, z_k], dim=1)
            alpha_k = self.gate_net(gate_input)
            S_k = alpha_k * z_pred + (1 - alpha_k) * s_pred
            S_hat_list.append(S_k)
        return torch.stack(S_hat_list, dim=1)  # [B, N, C, D, H, W]

# Specific Frequency-Aware Feature Extraction (SFAFE)
class SFAFE(nn.Module):
    """
    Disentangles frequency-specific (incremental) features from shared/base features.
    Uses two encoders and four cross-reconstruction decoders to encourage
    complementary representations; returns reconstruction loss and the
    task-specific latent ("var").
    """
    def __init__(self, diameters_len, f0_channels=1024, fused_channels=128, latent_com=256, latent_var=128):
        super().__init__()

        input_dim_f0 = f0_channels
        input_dim_fused = fused_channels * diameters_len

        # Encoders for base (f0) branch.
        self.encoder_com0 = nn.Sequential(
            nn.Linear(input_dim_f0, latent_com),
            nn.LayerNorm(latent_com),
            nn.ReLU(inplace=True)
        )
        self.encoder_var0 = nn.Sequential(
            nn.Linear(input_dim_f0, latent_var),
            nn.LayerNorm(latent_var),
            nn.ReLU(inplace=True)
        )
        # Map concatenated fused bands into a compact fused_channels.
        self.fused_mapper = nn.Sequential(
            nn.Linear(input_dim_fused, fused_channels),
            nn.LayerNorm(fused_channels),
            nn.ReLU(inplace=True)
        )
        # Encoders for fused (frequency) branch.
        self.encoder_com = nn.Sequential(
            nn.Linear(fused_channels, latent_com),
            nn.LayerNorm(latent_com),
            nn.ReLU(inplace=True)
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(fused_channels, latent_var),
            nn.LayerNorm(latent_var),
            nn.ReLU(inplace=True)
        )
        # Four decoders for cross/self reconstructions.
        self.decoder_f0_from_fused = nn.Sequential(
            nn.Linear(latent_com + latent_var, f0_channels),
            nn.LayerNorm(f0_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder_f0_from_f0 = nn.Sequential(
            nn.Linear(latent_com + latent_var, f0_channels),
            nn.LayerNorm(f0_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder_fused_from_fused = nn.Sequential(
            nn.Linear(latent_com + latent_var, fused_channels),
            nn.LayerNorm(fused_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder_fused_from_f0 = nn.Sequential(
            nn.Linear(latent_com + latent_var, fused_channels),
            nn.LayerNorm(fused_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f0, fused):
        """
        Args:
            f0:    [B, f0_channels] pooled base features from the student encoder.
            fused: [B, diameters_len * fused_channels] concatenated FSTE outputs.
        Returns:
            loss_recon: scalar reconstruction loss across four decoders.
            var:        [B, latent_var] task-specific (incremental) latent.
        """
        # Map fused multi-band features.
        fused_feat = self.fused_mapper(fused)    # [B, fused_channels]

        # Encode both branches.
        com0 = self.encoder_com0(f0)               # [B, latent_com]
        var0 = self.encoder_var0(f0)               # [B, latent_var]
        com = self.encoder_com(fused_feat)         # [B, latent_com]
        var = self.encoder_var(fused_feat)         # [B, latent_var]

        # Four reconstructions: two to f0 and two to fused_feat.
        recon_f0_from_fused = self.decoder_f0_from_fused(torch.cat([com, var0], dim=1))
        recon_f0_from_f0    = self.decoder_f0_from_f0(torch.cat([com0, var0], dim=1))
        recon_fused_from_fused = self.decoder_fused_from_fused(torch.cat([com, var], dim=1))
        recon_fused_from_f0    = self.decoder_fused_from_f0(torch.cat([com0, var], dim=1))

        # Reconstruction losses (MSE).
        loss_recon1 = F.mse_loss(recon_f0_from_fused, f0)
        loss_recon2 = F.mse_loss(recon_f0_from_f0,    f0)
        loss_recon3 = F.mse_loss(recon_fused_from_fused, fused_feat)
        loss_recon4 = F.mse_loss(recon_fused_from_f0,    fused_feat)

        # Aggregate (uniform weights by default).
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3 + loss_recon4) / 4
        return loss_recon, var

    def get_dec_features(self, fused):
        # At inference: only extract the task-specific latent from the fused branch.
        fused_feat = self.fused_mapper(fused)
        var = self.encoder_var(fused_feat)
        return var


class FADRNet(nn.Module):
    """
    Frequency-Aware Dual-Retention Network (FADRNet):
    - Student base encoder for old tasks
    - FFA -> FSTE to build incremental task-specific frequency spaces
    - SFAFE to distill task-specific increments
    - DRM to preserve old-task knowledge via output KD and classifier-structure alignment
    """
    def __init__(self, basic_task, incre_task, diameters_len, in_channels=1, n_classes=2, down_channels=256):
        super(FADRNet, self).__init__()
        self.name = 'FADRNet_' + basic_task + '_' + incre_task
        self.student_model = PreIncrNet(basic_task=basic_task, in_channels=in_channels, n_classes=n_classes)
        # New-task classifier head over concatenated [var (uniq_incre), base features].
        self.student_model.CLSmodel_new = nn.Linear(1024, n_classes)
        self.student_model.fc = nn.Linear(1024 + down_channels, 1024)

        self.DRM = DRM(n_classes=n_classes)
        self.FFA = FFA(diameters_len=diameters_len, num_channels=1024, down_channels=down_channels)
        self.FSTE = FSTE(diameters_len=diameters_len, num_channels=down_channels)
        self.SFAFE = SFAFE(
            diameters_len=diameters_len, f0_channels=1024, fused_channels=down_channels,
            latent_com=down_channels//4, latent_var=down_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _set_requires_grad(self, nets, requires_grad=False):
        # Utility to (un)freeze one or a list of modules.
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def flatten(self, features):
        # Global average pooling and flatten.
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features

    def forward(self, x_list, label, loss_ce, teacher_model):
        """
        Args:
            x_list: list of length N; each item [B, 1, D, H, W] is the spatial subvolume
                    for a specific frequency band (pre-extracted outside or via masking).
            label:  [B] integer class labels of the incremental task.
            loss_ce: cross-entropy criterion for the incremental head.
            teacher_model: frozen teacher (previous stage) network.
        Returns:
            total_loss: scalar total loss (task + FSTE + SFAFE + DRM)
            logits_new: [B, C] logits from the incremental head.
        """
        B = x_list[0].shape[0]
        # Extract student backbone features for all frequency inputs at once.
        x_list_in = torch.cat(x_list, 0)                 # [N*B, 1, D, H, W]
        features_list = self.get_features(x_list_in)      # [N*B, C, D', H', W']
        _, C_, W_, H_, L_ = features_list.shape
        features_list = features_list.view(-1, B, C_, W_, H_, L_)  # [N, B, C, D', H', W']

        # Base features for the last band (or designated base path).
        features_old = self.flatten(features_list[-1])    # [B, 1024]

        # DRM: KD on current inputs; structure alignment via OT.
        loss_DRM = self.DRM(x_list[-1], features_old, self.student_model, teacher_model)

        # FFA -> FSTE to build frequency-aware fused features.
        fused = self.FFA(features_list)                   # [N, B, C, D', H', W']
        fused, loss_FSTE = self.FSTE(fused, train_flag=True)  # fused: [B, N, C, D', H', W']
        fused = self.flatten(fused)                       # [B, N*C]

        # SFAFE disentanglement using pooled features.
        loss_SFAFE, var = self.SFAFE(features_old, fused) # var: [B, 512]

        # Concatenate task-specific var with base features and classify for new task.
        features = torch.cat((var, features_old), dim=1)  # [B, 512+1024]
        new_features = self.student_model.fc(features)
        logits_new = self.student_model.CLSmodel_new(new_features)
        loss = loss_ce(logits_new, label)
        return loss + loss_FSTE + loss_SFAFE + loss_DRM, logits_new

    def get_features(self, x):
        # Student backbone features (no pooling).
        layer0 = self.student_model.backbone.layer0(x)
        layer1 = self.student_model.backbone.layer1(layer0)
        layer2 = self.student_model.backbone.layer2(layer1)
        layer3 = self.student_model.backbone.layer3(layer2)
        layer4 = self.student_model.backbone.layer4(layer3)
        return layer4

    def predictcls_old(self, x):
        # Inference logits for the base (old) head.
        with torch.no_grad():
            layer0 = self.student_model.backbone.layer0(x[-1])
            layer1 = self.student_model.backbone.layer1(layer0)
            layer2 = self.student_model.backbone.layer2(layer1)
            layer3 = self.student_model.backbone.layer3(layer2)
            layer4 = self.student_model.backbone.layer4(layer3)
            features = self.flatten(layer4)
            logits = self.student_model.CLSmodel(features)
            return logits

    def predictcls(self, x_list):
        # Inference logits for the incremental (new) head.
        with torch.no_grad():
            B = x_list[0].shape[0]
            x_list_in = torch.cat(x_list, 0)
            features_list = self.get_features(x_list_in)
            _, C_, W_, H_, L_ = features_list.shape
            features_list = features_list.view(-1, B, C_, W_, H_, L_)

            features_old = self.flatten(features_list[-1])

            fused = self.FFA(features_list)
            fused = self.FSTE(fused, train_flag=False)    # [B, N, C, D', H', W']
            fused = self.flatten(fused)
            var = self.SFAFE.get_dec_features(fused)

            features = torch.cat((var, features_old), dim=1)
            new_features = self.student_model.fc(features)
            logits_new = self.student_model.CLSmodel_new(new_features)
            return logits_new


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from torch.nn import CrossEntropyLoss
    loss_Cross = CrossEntropyLoss(reduction='mean').cuda()

    diameters_list = [70, 110, 150, 999]
    batch_size = 1

    diameters_len = len(diameters_list)
    x_list = []
    for i in range(len(diameters_list)):
        x_list.append(torch.randn(batch_size, 1, 160, 192, 160).cuda())

    labels = torch.randint(0, 2, (batch_size,)).cuda()

    teacher_model = PreIncrNet(basic_task='1p19q', in_channels=1, n_classes=2).cuda()
    model = FADRNet(basic_task='1p19q', incre_task='IDH', diameters_len=diameters_len, in_channels=1, n_classes=2).cuda()
    # Copy teacher params into student (load matching keys only).
    teacher_state = teacher_model.state_dict()
    student_state = model.student_model.state_dict()
    filtered_state = {k: v for k, v in teacher_state.items() if k in student_state}
    student_state.update(filtered_state)
    model.student_model.load_state_dict(student_state)

    # Freeze teacher, train student.
    model._set_requires_grad(teacher_model, False)
    model._set_requires_grad(model.student_model, True)

    # Forward returns (total_loss, logits_new); unpack accordingly.
    total_loss, logits = model(x_list, labels, loss_Cross, teacher_model)
    print(total_loss.item(), logits.shape)