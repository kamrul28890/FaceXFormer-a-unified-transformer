# 3. The FaceXFormer Architecture

This section describes FaceXFormer as claimed in the paper and annotates divergences found in the released/reproduced code. We use three labels throughout: **(Paper)** for the original paper description, **(Repo)** for the released or current repository implementation, and **(Our impl.)** for the reproduced training pipeline used in this report.

## 3.1 Overall Framework

**(Paper)** FaceXFormer is an encoder-decoder transformer for multi-task facial analysis. The input is a 224x224 RGB face image. A Swin-B encoder extracts hierarchical multi-scale features at four resolutions. These features are projected and fused by an MLP-Fusion module into a unified face representation \(F\). The FaceX decoder then jointly processes the face representation and learnable task tokens \(T\), producing refined face tokens \(\hat{F}\) and refined task tokens \(\hat{T}\). Finally, a unified head applies another task-to-face attention step before routing the refined tokens into task-specific prediction heads.

**(Repo / Our impl.)** The reproduced repository follows this high-level pipeline: Swin-B backbone, multi-scale fusion to a 256-dimensional representation, a two-block FaceX-style decoder, a final token-to-image attention layer, and task-specific heads for eight tasks. The implementation returns outputs for landmark detection, head pose, attributes, visibility, age, gender, race, and segmentation. Facial expression recognition and face recognition are not implemented in the released/reproduced code path.

**Figure 3.1.** FaceXFormer end-to-end pipeline. Input image is processed by a Swin-B encoder, fused via MLP-Fusion, then jointly decoded by the FaceX decoder operating on both face tokens and learnable task tokens.
Source asset: `report_assets/fig1_facexformer_pipeline.pdf`.

**Divergence Note.** The high-level architecture is preserved for the eight reproduced tasks. The scope diverges from the paper because expression recognition and face recognition are absent from the implementation used here.

## 3.2 Encoder: Swin-B Backbone

**(Paper)** The encoder is an ImageNet-pretrained Swin-B transformer. It extracts four stages of hierarchical features, corresponding to progressively coarser spatial strides. The paper emphasizes that FaceXFormer does not use face-specific pretraining; the unified face representation is learned through multi-task co-training from an ImageNet-initialized backbone. This is an important design claim because it separates FaceXFormer from approaches that depend on large face-pretrained representation models.

**(Repo / Our impl.)** The code uses `torchvision.models.swin_b` as the backbone and projects the four encoder blocks into a shared decoder hidden size. The reproduced configuration uses a decoder embedding dimension of 256 and a decoder depth of 2.

**Divergence Note.** No major backbone divergence was identified. The relevant implementation question is not the backbone family, but the downstream token and head design.

## 3.3 MLP-Fusion Module

**(Paper)** The MLP-Fusion module follows the lightweight decoder-head design style used in SegFormer. Each of the four multi-scale feature maps is projected into a common target dimension, the projected maps are concatenated, and a final MLP fusion operation produces the unified face representation \(F\). The paper reports this module as lightweight, with approximately 983K parameters, and positions it as a key reason the model can use multi-scale information without the cost of a heavy pixel decoder.

**(Repo / Our impl.)** The reproduced model follows the same conceptual pattern: four Swin features are projected to a common 256-channel space, concatenated, and fused by a convolutional/MLP-style fusion block before being passed to the FaceX decoder.

**Divergence Note.** No major fusion-module divergence was found at the level needed for this report. The paper-level claim and the repository implementation agree on multi-scale projection, concatenation, and lightweight fusion.

## 3.4 Task Tokens

**(Paper)** FaceXFormer represents tasks using learnable task tokens. The paper explicitly states that segmentation uses one token per segmentation class, landmark prediction uses 68 tokens, and head pose uses 9 tokens corresponding to a 3x3 rotation matrix. It then states that one token is used for each of the other tasks. This matters because token count determines how much task-specific structure can be represented before the prediction head.

**(Repo / Our impl.)** The current code uses 18 task tokens total: 1 landmark token, 1 pose token, 1 attribute token, 1 visibility token, 1 age token, 1 gender token, 1 race token, and 11 segmentation mask tokens. The landmark token is passed through an MLP to output 136 coordinate values. The pose token is passed through an MLP to output 3 Euler-angle values. Attribute prediction uses one token and outputs 40 logits.

**Table 3.1: Task Token Counts - Paper vs. Repo / Our Implementation**

| Task | Paper Token Count | Repo / Our Implementation | Divergence |
| --- | ---: | ---: | --- |
| Face Parsing / Segmentation | One per segmentation class; paper appendix mentions 19 CelebAMask-HQ classes | 11 mask tokens | **MED - class mapping/token count ambiguity** |
| Landmark Detection | 68 tokens, one per keypoint | 1 token -> MLP -> 136 coordinates | **HIGH - fundamentally different tokenization** |
| Head Pose Estimation | 9 tokens for a 3x3 rotation matrix | 1 token -> MLP -> 3 Euler angles | **HIGH - different representation and output geometry** |
| Attribute Prediction | 1 token; MLP outputs 40 attributes | 1 token -> MLP -> 40 logits | None confirmed |
| Age Estimation | 1 token | 1 token -> MLP -> 8 age bins | Minor output-bin implementation detail |
| Gender Classification | 1 token | 1 token -> MLP -> 2 logits | None |
| Race Classification | 1 token | 1 token -> MLP -> 5 logits | Minor class-taxonomy limitation |
| Face Visibility | 1 token | 1 token -> MLP -> 29 logits | None confirmed |
| Expression Recognition | 1 token | Not implemented | **HIGH - dropped task** |
| Face Recognition | 1 token | Not implemented | **HIGH - dropped task** |

*Caption: Task token counts as described in the paper compared with the released/reproduced implementation. The landmark and head-pose token divergences are the most consequential architecture-level mismatches.*

The landmark token discrepancy is especially important. A 68-token design can assign separate attention capacity to each landmark, allowing per-keypoint representations to interact with the face tokens and with each other. Collapsing all landmarks into a single token compresses the geometric signal before prediction. Even if the final MLP still emits 136 coordinate values, the attention mechanism no longer has one learned query per landmark. This likely weakens the landmark task as a source of spatially precise gradients for the shared face representation.

**Divergence Note.** The original outline stated that the paper used 40 attribute tokens. Cross-checking the paper text indicates otherwise: the paper says one token is used for other tasks after segmentation, landmarks, and head pose. Attribute prediction still outputs 40 binary labels, but those labels are produced from a single task token.

## 3.5 FaceX Decoder

**(Paper)** The FaceX decoder is the central architectural novelty. It uses \(N=2\) decoder blocks. Each block models interactions among task tokens and face tokens through three attention operations:

- **Task Self-Attention (TSA):** task tokens attend to other task tokens, allowing task relationships to be modeled explicitly.
- **Task-to-Face Cross-Attention (TFCA):** task tokens act as queries while face tokens act as keys and values, allowing each task to extract visual evidence from the fused face representation.
- **Face-to-Task Cross-Attention (FTCA):** face tokens act as queries while task tokens act as keys and values, allowing the shared face representation to be refined by task-specific information.

The bidirectional aspect is the key difference from standard task-to-image decoders. In a standard cross-attention design, task tokens can read from the image representation, but the image representation itself is not updated by the task tokens. FaceXFormer adds the reverse direction through FTCA. The paper argues that this feedback loop improves the shared face representation, especially for tasks that depend on spatially detailed face tokens.

**(Repo / Our impl.)** The `TwoWayAttentionBlock` implementation matches the core paper structure: task self-attention, token-to-image cross-attention, an MLP block, and image-to-token cross-attention. The `TwoWayTransformer` stacks two such blocks and then applies a final token-to-image attention layer corresponding to the unified head. This is structurally consistent with the paper's two-block FaceX decoder description.

**Figure 3.2.** FaceX decoder block detail showing TSA, TFCA, and FTCA.
Source asset: `report_assets/fig2_facex_decoder_block.pdf`.

**Divergence Note.** No major decoder-operation divergence was found. The repository implements the bidirectional attention pattern that the paper claims. The larger discrepancies occur in token counts, task scope, and prediction heads.

## 3.6 Unified Head and Task-Specific Prediction Heads

**(Paper)** The unified head applies a final task-to-face cross-attention operation before task-specific prediction. The paper describes an hourglass network for landmark prediction, a regression MLP for head pose, PartialFC with ArcFace for face recognition, an upsampling and cross-product mechanism for segmentation, and classification or regression MLPs for the remaining tasks.

**(Repo / Our impl.)** The implementation uses a final token-to-image attention layer and then routes each token to lightweight heads. Segmentation uses 11 mask tokens, an upscaling module, and a hypernetwork-style cross-product with upscaled face features. Landmark prediction uses a single-token MLP outputting 136 coordinates. Head pose uses a single-token MLP outputting 3 Euler angles. Attributes, visibility, age, gender, and race use MLP heads. Expression recognition and face recognition heads are absent.

**Table 3.2: Task Prediction Heads - Paper vs. Repo / Our Implementation**

| Task | Paper Head | Repo / Our Implementation | Loss Function in Reproduction | Divergence |
| --- | --- | --- | --- | --- |
| Segmentation | Upsample + cross-product with segmentation tokens | Implemented with 11 mask tokens and upscaled face features | Dice + CE | Class mapping/token count ambiguity |
| Landmark | Hourglass network on 68 landmark tokens | MLP on 1 landmark token -> 136 coordinates | STARLoss-style landmark loss | **HIGH** |
| Head Pose | Regression MLP from 9 rotation-matrix tokens | MLP on 1 pose token -> 3 Euler angles | Geodesic loss via Euler-to-rotation conversion | **HIGH** |
| Attribute | Classification MLP | MLP on 1 token -> 40 logits | BCE with logits | None confirmed |
| Age | Classification/regression MLP | MLP on 1 token -> 8 age-bin logits | L1 + CE over age bins | Minor bin-boundary detail |
| Gender | Classification MLP | MLP on 1 token -> 2 logits | CE | None |
| Race | Classification MLP | MLP on 1 token -> 5 logits | CE | Class taxonomy differs from FairFace 7-class labels |
| Visibility | Classification MLP | MLP on 1 token -> 29 logits | BCE with logits | None confirmed |
| Expression | Classification MLP | Not implemented | - | Dropped |
| Face Recognition | PartialFC + ArcFace | Not implemented | - | Dropped |

*Caption: Task-specific prediction heads and loss functions. The landmark and head-pose heads are the largest implementation divergences relative to the paper description.*

**Divergence Note.** The landmark head is the most consequential head-level mismatch: the paper describes an hourglass network operating after a 68-token landmark representation, whereas the reproduced implementation uses a single task token and a direct MLP. Head pose is also materially different because the paper describes a 3x3 rotation-matrix representation, while the implementation predicts 3 Euler angles. These choices change not only the task output parameterization but also the gradients that flow back through the FaceX decoder into the shared representation.
