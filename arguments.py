import argparse
import os

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,
                choices=['baseline', 'custom', 'supcon'],
                help="baseline: fine-tune BERT for classification; "
                     "custom: advanced fine-tuning techniques; "
                     "supcon: contrastive learning (SupContrast / SimCLR)")

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int,
                help="number of layers that are reinitialized. Count from last to first.")

    # contrastive learning options [AI Assisted: Claude Code]
    parser.add_argument("--contrast-loss", default="supcon", type=str,
                choices=['supcon', 'simclr'],
                help="which contrastive loss to use for --task supcon: "
                     "'supcon' (SupContrast, uses labels) or 'simclr' (unsupervised).")
    parser.add_argument("--temperature", default=0.07, type=float,
                help="temperature for the SupCon/SimCLR contrastive loss.")
    
    # Others
    parser.add_argument("--input-dir", default='assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--log-interval", default=500, type=int,
                help="Number of steps between logging (used by check_directories in debug mode).")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    # [AI Assisted: Claude Code] defaults tuned for BERT fine-tuning:
    # 2e-5 is in the standard 1e-5..5e-5 range for full BERT fine-tuning (1e-4 diverges),
    # hidden-dim 768 avoids bottlenecking the 768-d [CLS] embedding before 18 classes,
    # and drop-rate 0.1 matches BERT's own hidden dropout (0.9 destroys the signal).
    parser.add_argument("--learning-rate", default=2e-5, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=768, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.1, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")

    args = parser.parse_args()
    return args
