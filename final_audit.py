
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import sys
import os
import chess
from transformers import AutoModel
import multiprocessing as mp

class Encoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Encoder, self).__init__(); self.shared_layers = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(True), nn.Linear(256, 128), nn.ReLU(True))
        self.fc_mu = nn.Linear(128, bottleneck_dim); self.fc_log_var = nn.Linear(128, bottleneck_dim)
    def forward(self, x): hidden = self.shared_layers(x); return self.fc_mu(hidden), self.fc_log_var(hidden)

class Decoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(Decoder, self).__init__(); self.decoder_layers = nn.Sequential(nn.Linear(bottleneck_dim, 256), nn.ReLU(True), nn.Linear(256, input_dim))
    def forward(self, z): return self.decoder_layers(z)

class ConceptEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=8):
        super(ConceptEncoder, self).__init__(); self.encoder = Encoder(input_dim, bottleneck_dim); self.decoder = Decoder(input_dim, bottleneck_dim)
    def reparameterize(self, mu, log_var): std = torch.exp(0.5 * log_var); epsilon = torch.randn_like(std); return mu + std * epsilon
    def forward(self, x): mu, log_var = self.encoder(x); z = self.reparameterize(mu, log_var); return self.decoder(z), mu, log_var

class Hook:
    def __init__(self): self.value = None
    def __call__(self, module, input, output): self.value = output.detach()

def fen_to_tensor(fen: str):
    board = chess.Board(fen); P = 19; tensor = np.zeros((8, 8, P), dtype=np.float32)
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8); plane = piece_map[piece.symbol()]; tensor[7 - rank, file, plane] = 1.0
    tensor[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    if board.ep_square is not None: rank, file = divmod(board.ep_square, 8); tensor[7 - rank, file, 13] = 1.0
    tensor[:, :, 14] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0; tensor[:, :, 15] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, 16] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0; tensor[:, :, 17] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    return tensor

class PuzzleDataset(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, idx): return self.df.iloc[idx]['FEN']

def collate_fn_puzzle(batch_fens: list):
    tensors = [fen_to_tensor(fen) for fen in batch_fens]
    return np.stack(tensors)[:, np.newaxis, :, :, :]


def main():
    parser = argparse.ArgumentParser(description="Final audit", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--puzzle_db', type=str, required=True, help="")
    parser.add_argument('--concept_encoder_ckpt', type=str, required=True, help="")
    parser.add_argument('--top_k', type=int, default=500, help="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running"); print(f"Using device: {device}")

    try:
        print("Loading and decompressing db")
        puzzles_df = pd.read_csv(args.puzzle_db, compression='zstd')
        print(f"Loaded {len(puzzles_df)} puzzles.")
    except Exception as e:
        print(f"Could not load the puzzle database", file=sys.stderr); print(f"Error: {e}", file=sys.stderr); sys.exit(1)

    print("Loading Maxlegrec/ChessBot")
    powercore_model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True).to(device).eval()
    thought_vector_hook = Hook()
    hook_handle = powercore_model.layers[-1].register_forward_hook(thought_vector_hook)

    concept_encoder = ConceptEncoder(input_dim=512).to(device).eval()
    concept_encoder.load_state_dict(torch.load(args.concept_encoder_ckpt, map_location=device))
    
    puzzle_dataset = PuzzleDataset(puzzles_df)
    puzzle_loader = DataLoader(puzzle_dataset, batch_size=1024, shuffle=False, num_workers=4, collate_fn=collate_fn_puzzle, pin_memory=True)

    print("Generating 8-factor representations")
    puzzle_factors = []
    with torch.no_grad():
        for tensor_batch in tqdm(puzzle_loader, desc="Processing Puzzles"):
            inputs = torch.from_numpy(tensor_batch).to(device)
            powercore_model(inputs_embeds=inputs)
            captured_thought_vector = thought_vector_hook.value.mean(dim=1)
            mu, _ = concept_encoder.encoder(captured_thought_vector)
            puzzle_factors.append(mu.cpu().numpy())
    
    puzzle_factors = np.vstack(puzzle_factors)
    hook_handle.remove()
    
    # Add the learned factors to our main DataFrame
    for i in range(8):
        puzzles_df[f'factor_{i+1}'] = puzzle_factors[:, i]
    
    puzzles_df['ThemeList'] = puzzles_df['Themes'].str.split()
    
    print("\n" + "="*80)
    print("results")
    print("="*80)
    
    for i in range(8):
        factor_name = f'factor_{i+1}'
        print(f"\n--- Profiling {factor_name.upper()} ---")

        top_k_df = puzzles_df.nlargest(args.top_k, factor_name)
        theme_counts = top_k_df['ThemeList'].explode().value_counts(normalize=True)
        
        print(f"Top {args.top_k} highest-activation positions for {factor_name} are composed of the following themes:")
        for theme, percentage in theme_counts.head(5).items():
            print(f"  - {percentage:7.2%} : {theme}")
        

if __name__ == "__main__":
    if sys.platform.startswith("linux") or sys.platform == "darwin": mp.set_start_method('spawn', force=True)
    main()