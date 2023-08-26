from modal import Image, Stub, method

stub = Stub("stable-diffusion-cli")


def download_models():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "accelerate",
        "diffusers[torch]>=0.15.1",
        "ftfy",
        "torchvision",
        "transformers~=4.25.1",
        "triton",
        "safetensors",
    )
    .pip_install(
        "torch==2.0.1+cu117",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .pip_install("xformers", pre=True)
    .pip_install("transformers==4.32.0")
    .run_function(download_models)
)
stub.image = image

@stub.cls(gpu="A10G")
class StableDiffusion:
    def __enter__(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    @method()
    def run_inference(
        self, img_url: str = None
    ) -> list[bytes]:
        import requests
        from PIL import Image
        
        img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

        # conditional image captioning
        text = "a photography of"
        inputs = self.processor(raw_image, text, return_tensors="pt").to("cuda")

        out = self.model.generate(**inputs)
        # print(processor.decode(out[0], skip_special_tokens=True))
        # >>> a photography of a woman and her dog

        # unconditional image captioning
        inputs = self.processor(raw_image, return_tensors="pt").to("cuda")

        out = self.model.generate(**inputs)
        print(self.processor.decode(out[0], skip_special_tokens=True))