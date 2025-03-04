import argparse
import torch

from models.counting_guidance_pipeline import CountingGuidancePipeline
from utils import ptp_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, type=str, help="Prompt to generate image from")
    parser.add_argument("--output", type=str, help="Output file path", default="output.png")
    parser.add_argument("--model", type=str, help="Model path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--counting_words", type=str, nargs="+", help="Word(s) to apply counting loss to.", required=True)
    parser.add_argument("--counting_word_counts", type=int, nargs="+", help="Cound for each word", required=True)
    parser.add_argument("--counting_scale", type=float, nargs="+", help="Counting loss scale. Specify two values for linear interpolation", default=[1.0])
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    gen = torch.Generator('cuda').manual_seed(args.seed)
    model = CountingGuidancePipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

    controller = ptp_utils.AttentionStore()
    ptp_utils.register_attention_control(model, controller)

    words = args.prompt.split(" ")
    indices_to_alter = [words.index(word) + 1 for word in args.counting_words]

    counting_scale = args.counting_scale

    if len(counting_scale) == 1:
        counting_scale = counting_scale * len(args.counting_words)

    outputs = model(
        prompt=args.prompt,
        attention_store=controller,
        indices_to_alter=indices_to_alter,
        generator=gen,
        max_iter_to_alter=9999,
        thresholds=None,
        # scale_factor=20,
        # scale_range=(1., 0.5),

        scale_factor=None,
        scale_range=None,

        sigma=0.5,
        kernel_size=3,
        sd_2_1=False,
        save_imgs=False,

        attention_loss_start=None,
        attention_loss_scale=None,

        # token_counts=[2],
        token_counts=args.counting_word_counts,
        counting_loss_scales=counting_scale,
    )
    image = outputs.images[0]

    image.save(args.output)


if __name__ == "__main__":
    main()
