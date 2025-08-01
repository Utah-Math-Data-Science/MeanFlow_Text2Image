# Multimodal Meanflow: Text2Image

This repository implements a one-step generative Flow Matching model that combines [Flowtok](https://arxiv.org/pdf/2503.10772) and [MeanFlow](https://arxiv.org/abs/2505.13447).
## Framework
    Texts -> Token Embedding -> Latent Variable $Z_0$ 
            ------ Meanflow ------> 
    Latent Variable $Z_1$ -> VAE.decode($Z_1$) -> Images
## Prompts:
    1) a yellow common dandelion in the middle of a yellow flower
    2) an orange dahlia in the garden
    3) a red bromelia in a plant
    4) a pink cape flower in the garden
    5) a close-up of a pink cyclamen
    6) a red petunia in the garden
    7) a purple pincushion flower in the grass
    8) a gazania in a flower bed in the garden
    9) a yellow sunflower in the middle of a field
    10) two orange English marigolds in a black background
    11) a small pink cyclamen in the middle of a flower
    12) a morning glory in a fenced-in garden
    
## One-step Generated Samples:
![Samples](Samples.png)

## Pretrained Models:
### Image to latent space:
```
from diffusers.models import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
```
### Prompts to latent space:
```
from transformers import AutoTokenizer, AutoModel
pre_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
pre_model = AutoModel.from_pretrained("intfloat/e5-base")
```


