import argparse
import torch
import torch.nn.functional as F

from model import BengioLM


def load_vocab(vocab_path):
    d = torch.load(vocab_path)
    return d['stoi'], d['itos']


def generate_text(model, stoi, itos, seed, length=500, temperature=1.0, top_k=None, device='cpu'):
    model.eval()
    context_len = model.k
    # prepare context indices
    context = [stoi.get(c, None) for c in seed]
    if any(v is None for v in context):
        # if seed contains unknown char, replace with first token (safe fallback)
        context = [stoi[c] if c in stoi else 0 for c in seed]

    if len(context) < context_len:
        context = [0] * (context_len - len(context)) + context
    context = context[-context_len:]

    generated = seed
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([context], dtype=torch.long, device=device)  # [1, k]
            logits = model(x)[0] / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                values, indices = torch.topk(probs, top_k)
                values = values / values.sum()
                next_idx = indices[torch.multinomial(values, 1)[0]].item()
            else:
                next_idx = torch.multinomial(probs, 1).item()

            generated += itos[next_idx]
            context = context[1:] + [next_idx]

    return generated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vocab', type=str, default=None)
    parser.add_argument('--seed', type=str, default='\n')
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)

    # load vocab and args
    if args.vocab is not None:
        stoi, itos = load_vocab(args.vocab)
    else:
        stoi = ckpt.get('stoi')
        itos = ckpt.get('itos')

    model_args = ckpt.get('args', {})
    model = BengioLM(vocab_size=len(stoi), context_len=model_args.get('context_len', 10), embed_dim=model_args.get('embed_dim', 32), hidden_dim=model_args.get('hidden_dim', 128))
    model.load_state_dict(ckpt['model_state'])
    model.to(args.device)

    text = generate_text(model, stoi, itos, seed=args.seed, length=args.length, temperature=args.temperature, top_k=args.top_k, device=args.device)
    print(text)