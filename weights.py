from transformers import AutoModelForCausalLM
import torch
import argparse
from pathlib import Path


def get_weights():
    parser = argparse.ArgumentParser(description="Retrieve Huggingface Deepseek coder weights")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct", torch_dtype="auto", trust_remote_code=True
    )
    parser.add_argument("--path",
                        type=str,
                        default="weights",
                        help="The path to save model weights.", )
    args = parser.parse_args()
    path = Path(args.path)
    print("creating path....")
    path.mkdir(parents=True, exist_ok=True)
    print("saving weights....")
    torch.save(model.state_dict(), str(path / 'deepseek-weights.pt'))
    print("weights saved successfully in " + str(path / 'deepseek-weights.pt'))


if __name__ == '__main__':
    get_weights()
