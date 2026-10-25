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
from sklearn.metrics import roc_auc_score
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
    parser = argparse.ArgumentParser(description="Run the Definitive Conceptual Purity Audit.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--puzzle_db', type=str, required=True, help="Path to the lichess_db_puzzle.csv.zst file.")
    parser.add_argument('--concept_encoder_ckpt', type=str, required=True, help="Path to the trained ConceptEncoder checkpoint.")
    parser.add_argument('--num_puzzles', type=int, default=1000000, help="Number of puzzles to use for the audit.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"--- 🔬 Running Definitive Conceptual Purity Audit ---"); print(f"Using device: {device}")

    try:
        print("Loading and decompressing puzzle database..."); puzzles_df = pd.read_csv(args.puzzle_db, compression='zstd', nrows=args.num_puzzles)
        print(f"Loaded {len(puzzles_df)} puzzles.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load puzzle DB. Ensure 'zstandard' is installed.", file=sys.stderr); print(f"Error: {e}", file=sys.stderr); sys.exit(1)

    print("Loading PowerCore model (Maxlegrec/ChessBot)..."); powercore_model = AutoModel.from_pretrained("Maxlegrec/ChessBot", trust_remote_code=True).to(device).eval()
    thought_vector_hook = Hook(); hook_handle = powercore_model.layers[-1].register_forward_hook(thought_vector_hook)

    print("Loading trained ConceptEncoder..."); concept_encoder = ConceptEncoder(input_dim=512).to(device).eval()
    concept_encoder.load_state_dict(torch.load(args.concept_encoder_ckpt, map_location=device))
    
    puzzle_dataset = PuzzleDataset(puzzles_df); puzzle_loader = DataLoader(puzzle_dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=collate_fn_puzzle, pin_memory=True)

    print("Generating 8-factor representations for all puzzles...")
    puzzle_factors = []
    with torch.no_grad():
        for tensor_batch in tqdm(puzzle_loader, desc="Processing Puzzles"):
            inputs = torch.from_numpy(tensor_batch).to(device); powercore_model(inputs_embeds=inputs)
            captured_thought_vector = thought_vector_hook.value.mean(dim=1); mu, _ = concept_encoder.encoder(captured_thought_vector)
            puzzle_factors.append(mu.cpu().numpy())
    puzzle_factors = np.vstack(puzzle_factors); hook_handle.remove()
    
    print("\n--- Definitive Conceptual Purity Audit Results (AUC Score) ---")
    print("Measures how purely each factor represents a specific chess concept.")
    print("AUC > 0.9 = Revolutionary | AUC > 0.7 = Strong | AUC > 0.6 = Promising")
    print("-" * 80)
    
    puzzles_df['ThemeList'] = puzzles_df['Themes'].str.split()
    all_themes = puzzles_df['ThemeList'].explode().unique()
    # Exclude non-conceptual "meta" tags for a cleaner report
    meta_tags_to_exclude = {'mate', 'short', 'long', 'veryLong', 'oneMove', 'master', 'masterVsMaster', 'superGM'}
    themes_to_test = [theme for theme in all_themes if theme not in meta_tags_to_exclude]
    
    results = {}
    for theme in tqdm(themes_to_test, desc="Auditing Themes"):
        y = puzzles_df['ThemeList'].apply(lambda themes: theme in themes).astype(int)
        if len(y.unique()) < 2: continue
        theme_aucs = [max(roc_auc_score(y, puzzle_factors[:, i]), 1 - roc_auc_score(y, puzzle_factors[:, i])) for i in range(8)]
        results[theme] = theme_aucs

    report_df = pd.DataFrame(results, index=[f'factor_{i+1}' for i in range(8)]).T
    report_df['purity_score'] = report_df.max(axis=1)
    report_df['best_factor'] = report_df.idxmax(axis=1)
    
    # Set display options to show all rows
    pd.set_option('display.max_rows', None)
    print(report_df.sort_values('purity_score', ascending=False).round(4))
    
    print("\n--- Definitive Analysis ---")
    breakthroughs, strong_concepts = 0, 0
    for theme, row in report_df.sort_values('purity_score', ascending=False).iterrows():
        score, factor = row['purity_score'], row['best_factor']
        if score > 0.90:
            print(f"✅✅✅ REVOLUTION: Found a pure concept for '{theme}'. Factor '{factor}' is the definitive '{theme.upper()}' factor (AUC: {score:.4f}).")
            breakthroughs += 1
        elif score > 0.70:
            print(f"  ✅ Breakthrough: Found a strong concept for '{theme}'. Best predictor is '{factor}' (AUC: {score:.4f}).")
            strong_concepts += 1
            
    if breakthroughs == 0 and strong_concepts == 0:
        print("\nNo revolutionary concepts were discovered, but the audit has successfully quantified the conceptual knowledge of each factor.")
    
    print("\n--- Definitive Audit Complete ---")

if __name__ == "__main__":
    if sys.platform.startswith("linux") or sys.platform == "darwin": mp.set_start_method('spawn', force=True)
    main()