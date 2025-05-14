import argparse
import torch
from model.modeling_UniBioseq import UniBioseqModel

def main():
    parser = argparse.ArgumentParser(description="Merge model weights script")
    
    parser.add_argument("--heads_weights", type=str, required=True,
                        help="the path to the prediction head weights file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="the path to save the merged weights file")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading base model: BiooBang")
        model = UniBioseqModel.from_pretrained("lonelycrab88/BiooBang-1.0")
        model_weights = model.state_dict()
        
        print(f"loading the prediction head weights: {args.heads_weights}")
        heads_weights = torch.load(args.heads_weights)
        
        print("merging the weights...")
        finetuned_weights = {**model_weights, **heads_weights}
        
        print(f"saving the weights to: {args.output_path}")
        torch.save(finetuned_weights, args.output_path)
        
        print("Weight merging completed!")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An unknown error occurred: {e}")

if __name__ == "__main__":
    main()