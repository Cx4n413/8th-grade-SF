
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import argparse
from tqdm import tqdm
import sys
import os
import multiprocessing as mp

# thanks huggingface i love your library so much
from transformers import AutoModel

def fen_to_tensor(fen: str):
    board = chess.Board(fen)
    P = 19
    tensor = np.zeros((8, 8, P), dtype=np.float32)
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        plane = piece_map[piece.symbol()]
        tensor[7 - rank, file, plane] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[7 - rank, file, 13] = 1.0
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 15] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 16] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 17] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    return tensor

class Hook:
    def __init__(self): self.value = None
    def __call__(self, module, input, output): self.value = output.detach()

class FENDataset(Dataset):
    def __init__(self, fen_file_path):
        with open(fen_file_path, 'r') as f: self.fens = [line.strip() for line in f if line.strip()]
    def __len__(self): return len(self.fens)
    def __getitem__(self, idx): return self.fens[idx]

def collate_fn_chessbot(batch_fens: list):
    tensors = [fen_to_tensor(fen) for fen in batch_fens]
    return np.stack(tensors)[:, np.newaxis, :, :, :]


    
def main():
    parser = argparse.ArgumentParser(description="gen thought vectors", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fen_file', type=str, required=True, help="")
    parser.add_argument('--output_file', type=str, default="hf_rich_thought_vectors.pt", help="")
    parser.add_argument('--batch_size', type=int, default=256, help="uwu")
    parser.add_argument('--num_workers', type=int, default=8, help="num workers")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Rich Thought Vector Generation ---")
    print(f"Using device: {device}")
    model_name = "Maxlegrec/ChessBot"
    try:
        print(f"Downloading and loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        print("loaded huggingface model")
    except Exception as e:
        print(f"\nCouldnt load the Huggingface model.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    # -----------------------------------------------
        
    thought_vector_hook = Hook()
    hook_handle = model.layers[-1].register_forward_hook(thought_vector_hook)

    dataset = FENDataset(args.fen_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_chessbot, pin_memory=True)
    
    all_rich_vectors = []
    print(f"Processing {len(dataset)} positions...")
    for tensor_batch in tqdm(dataloader, desc="Generating Rich Vectors"):
        with torch.no_grad():
            inputs = torch.from_numpy(tensor_batch).to(device)
            model(inputs_embeds=inputs)
            
            raw_thought_tensor = thought_vector_hook.value
            mean_vec = raw_thought_tensor.mean(dim=1)
            std_vec = raw_thought_tensor.std(dim=1)
            max_vec = torch.max(raw_thought_tensor, dim=1).values
            
            rich_vector = torch.cat((mean_vec, std_vec, max_vec), dim=1)
            all_rich_vectors.append(rich_vector.cpu())

    hook_handle.remove()
    final_tensor = torch.cat(all_rich_vectors, dim=0)
    torch.save(final_tensor, args.output_file)

    print(f"Output file saved to: {args.output_file}")

if __name__ == "__main__":
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        mp.set_start_method('spawn', force=True)
    main()