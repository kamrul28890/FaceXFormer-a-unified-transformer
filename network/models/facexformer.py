import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Any, Optional, Tuple, Type
from torchvision.models import swin_b
from .transformer import TwoWayTransformer, LayerNorm2d

class MLP(nn.Module):  # regular MLP
    # 3-layer MLP: 256 → 256 → 256 → output_dim
    # ReLU activation between layers
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
class FaceDecoder(nn.Module):
    # Decoder along with task-specific heads
    def __init__(
        self,
        *,
        transformer_dim: 256,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # defining total 18 task tokens; each token is a learnable 256-D vector initialized randomly
        self.landmarks_token = nn.Embedding(1, 256)    # 1 token for landmarks
        self.pose_token = nn.Embedding(1, 256)         # 1 token for pose
        self.attribute_token = nn.Embedding(1, 256)    # 1 token for attributes
        self.visibility_token = nn.Embedding(1, 256)   # 1 token for visibility
        self.age_token = nn.Embedding(1, 256)          # 1 token for age
        self.gender_token = nn.Embedding(1, 256)       # 1 token for gender
        self.race_token = nn.Embedding(1, 256)         # 1 token for race
        self.mask_tokens = nn.Embedding(11, 256)       # 11 tokens for segmentation (11 classes)
        
        # used only to upsample the segmentation mask
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2), # 56×56 → 112×112
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), # 112×112 → 224×224
            activation(),
        )
        # [B, 256, 56, 56] → [B, 32, 224, 224]

        # First ConvTranspose2d: [B, 256, 56, 56] → [B, 64, 112, 112]
        # Reduces channels: 256 → 64
        # Doubles spatial resolution: 56×56 → 112×112

        # Second ConvTranspose2d: [B, 64, 112, 112] → [B, 32, 224, 224]
        # Reduces channels: 64 → 32
        # Doubles spatial resolution: 112×112 → 224×224
        
        self.output_hypernetwork_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
            )  # [B, 11, 256] → [B, 11, 32]
                
        # now defining individual task heads
        # No activation on final layer (raw logits)
        # Loss functions apply sigmoid/softmax as needed
        self.landmarks_prediction_head = MLP(
            transformer_dim, transformer_dim, 136, 3
        ) # 68 landmarks × 2 coords
        self.pose_prediction_head = MLP(
            transformer_dim, transformer_dim, 3, 3
        ) # yaw, pitch, roll
        self.attribute_prediction_head = MLP(
            transformer_dim, transformer_dim, 40, 3
        ) # 40 binary attributes
        self.visibility_prediction_head = MLP(
            transformer_dim, transformer_dim, 29, 3
        ) # 29 visibility points
        self.age_prediction_head = MLP(
            transformer_dim, transformer_dim, 8, 3
        )  # 8 buckets for age (0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
        self.gender_prediction_head = MLP(
            transformer_dim, transformer_dim, 2, 3
        ) # male/female
        self.race_prediction_head = MLP(
            transformer_dim, transformer_dim, 5, 3
        ) # 5 race classes (White, Black, Indian, Asian, Others)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.landmarks_token.weight, self.pose_token.weight, self.attribute_token.weight, self.visibility_token.weight, self.age_token.weight, self.gender_token.weight, self.race_token.weight,self.mask_tokens.weight], dim=0)
        # Tokens are concatenated into [B, 18, 256] tensor and fed to transformer decoder
         
        tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)

        src = image_embeddings
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
    
        landmarks_token_out = hs[:, 0, :]
        pose_token_out =  hs[:, 1, :]
        attribute_token_out = hs[:, 2, :]
        visibility_token_out = hs[:, 3, :]
        age_token_out = hs[:, 4, :]
        gender_token_out = hs[:, 5, :]
        race_token_out = hs[:, 6, :]
        mask_token_out =  hs[:, 7:, :]  # [B, 11, 256]
        # 11 mask tokens (one per segmentation class)
        # 256-D embedding per token
        
        
        landmark_output = self.landmarks_prediction_head(landmarks_token_out)
        headpose_output = self.pose_prediction_head(pose_token_out)
        attribute_output = self.attribute_prediction_head(attribute_token_out)
        visibility_output = self.visibility_prediction_head(visibility_token_out)
        age_output = self.age_prediction_head(age_token_out)
        gender_output = self.gender_prediction_head(gender_token_out)
        race_output = self.race_prediction_head(race_token_out)
        
        # for segmentation output
        # b=B, c=256, h=56, w=56
        src = src.transpose(1, 2).view(b, c, h, w) # [B, 3136, 256] → [B, 256, 3136] → [B, 256, 56, 56]
        upscaled_embedding = self.output_upscaling(src)   # [B, 256, 56, 56] → [B, 32, 224, 224]
        hyper_in = self.output_hypernetwork_mlps(mask_token_out) # [B, 11, 256] → [B, 11, 32]
        b, c, h, w = upscaled_embedding.shape  # b=B, c=32, h=224, w=224
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # [B, 11, 32] @ [B, 32, 50176] → [B, 11, 50176] → [B, 11, 224, 224]
        
        return landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Uses sinusoidal positional encoding with random Gaussian frequencies
    Produces [256, 56, 56] positional encoding tensor, which is added to image features before transformer decoder
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class FaceXFormerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)  # only one linear layer, Project to 256-D

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)   # [B, C, H, W] → [B, H*W, C]
        hidden_states = self.proj(hidden_states) # [B, H*W, 256]
        return hidden_states

class FaceXFormer(nn.Module):
    def __init__(self):
        super(FaceXFormer, self).__init__()

        swin_v2 = swin_b(weights='IMAGENET1K_V1')
        self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
        self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
        self.multi_scale_features = []
        

        embed_dim = 1024
        out_chans = 256
        
        self.pe_layer = PositionEmbeddingRandom(out_chans // 2)   

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name)) # Forward hooks intercept outputs from each Swin stage
        
        self.face_decoder = FaceDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,  # 2 transformer blocks (N=2)
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,  # 8 attention heads
            ))    
        
        num_encoder_blocks = 4
        hidden_sizes = [128, 256, 512, 1024]
        decoder_hidden_size = 256
        
        mlps = []
        for i in range(num_encoder_blocks):  # 4 encoder blocks
            mlp = FaceXFormerMLP(input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)  # linear_c is a list of 4 MLPs

        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks, # Concatenated 4 scales, 256x4
            out_channels=decoder_hidden_size,   # Fused output, 256
            kernel_size=1,
            bias=False,
        )

        # Concatenates 4 feature maps along channel dimension: [B, 256, 56, 56] × 4 → [B, 1024, 56, 56]
        # 1×1 convolution fuses channels: [B, 1024, 56, 56] → [B, 256, 56, 56]
        # 1×1 conv: Performs channel-wise weighted fusion (learns which scale is important for each spatial location)
    
    def save_features_hook(self, name):
        def hook(module, input, output):
            self.multi_scale_features.append(output.permute(0,3,1,2).contiguous()) # Each feature map is rearranged from [B, H, W, C] → [B, C, H, W] format
        return hook

    def forward(self, x, labels, tasks):
        self.multi_scale_features.clear()
        
        _,_,h,w = x.shape
        features = self.backbone(x).squeeze()
        
        
        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
        
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)  # each MLP, out of the 4 MLPs
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=self.multi_scale_features[0].size()[2:], mode="bilinear", align_corners=False
            )  # same size or upsample to 56x56
            all_hidden_states += (encoder_hidden_state,)
        
        # Input: [B, 256, 56, 56]
        # Target: (56, 56) (same size)
        # No change: [B, 256, 56, 56] → [B, 256, 56, 56]

        # Input: [B, 256, 28, 28]
        # Upsample ×2: [B, 256, 28, 28] → [B, 256, 56, 56]

        # Input: [B, 256, 14, 14]
        # Upsample ×4: [B, 256, 14, 14] → [B, 256, 56, 56]

        # Input: [B, 256, 7, 7]
        # Upsample ×8: [B, 256, 7, 7] → [B, 256, 56, 56]


        # fusing features from multi-scale encoder (MLP fusion)
        fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1)) # Concatenates along channel dimension (256×4 = 1024 channels)
        image_pe = self.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
        
        landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = self.face_decoder(
                image_embeddings=fused_states,
                image_pe=image_pe
            )
        
        # For DDP consistency, always return all outputs regardless of task filtering
        # The loss function will handle which outputs to use based on available labels
        return landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output