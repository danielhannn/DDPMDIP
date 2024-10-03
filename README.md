# GAI_Project4
>   Guiding DIP Early Stopping with DDPM-inspired Supervision

This experiment involves using forward process output of DDPM model (noisy image sequence) as the training target for DIP model.
We use 10 different training stage to train DIP model, starting from the most corrupted image and gradually moving towards the clean target image.

The noise adding process can implement as below

```py
def add_noise(image, beta_schedule, t):
    noise = torch.randn_like(image)
    beta_t = beta_schedule[t]
    alpha_t = 1 - beta_t
    noisy_image = torch.sqrt(alpha_t) * image + torch.sqrt(1 - alpha_t) * noise
    return noisy_image, noise
```

![image](https://github.com/LunarrrHound/GAI_hw4/assets/70794772/790ae061-058a-4f27-b397-8366b887da34)

We expect this approach will improve the DIP model reconstruction quality.Since the model can learn the hierarchical representation of target image.

You can run main.ipynb directly to reproduce the experiment.

## Requirements
```
git clone https://github.com/LunarrrHound/GAI_hw4.git
```
```
pip install -r requirements.txt
```
## Report

With total training of 1000 iteration, we use linearly increased iteration and constant learning-rate for each stage.

###  Loss Reducion process

![image](https://github.com/LunarrrHound/GAI_hw4/assets/70794772/2481bb2c-23dd-400b-ad9a-effb12a8fb8e)

###  **PSNR Comparison**

![image](https://github.com/LunarrrHound/GAI_hw4/assets/70794772/05e97d00-2475-4d63-8ef9-017b648c2876)

| with DDPM-inspired supervision | without |
| ---- | ---- |
| **30.301893047737707** | **28.684494237210124** |

###  **Visual Result**

>  model output(left) target (right)

with DDPM-inspired supervision :
  
![image](https://github.com/LunarrrHound/GAI_hw4/assets/70794772/927792b0-2657-4b8b-b6b4-fb62f61d84f8)

without:

![image](https://github.com/LunarrrHound/GAI_hw4/assets/70794772/bb03de5a-c2fe-4db8-95ef-c0042dd1e962)

###  **Conclusion**

Under the same total iterations, we got a better result on hierarchical training process.

With PSNR evaluation improved from 28.684 to 30.301 and significant improvements on output image quality, we can say that this approach can increase learning efficiency of DIP model.




