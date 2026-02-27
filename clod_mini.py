"""
clod_mini.py — a micro Claude

A minimal implementation of how Claude works, in pure dependency-free Python.

The difference between a plain language model and Claude is three things:

  1. CHAT FORMAT    Conversations are structured with role tokens [H] and [A].
                   The model learns that [H] tokens are input, [A] tokens are output.

  2. SFT           Supervised Fine-Tuning. Loss is computed only on assistant tokens,
                   not human tokens. The model learns to respond, not just predict.

  3. DPO           Direct Preference Optimization. Given pairs of (chosen, rejected)
                   responses, the model learns to prefer the better one without needing
                   a reward model or PPO. The loss is:
                   -log σ( β · [ log π_θ(chosen) - log π_ref(chosen)
                                - log π_θ(rejected) + log π_ref(rejected) ] )

  + CONSTITUTIONAL  At inference: generate a draft, critique it, revise it.
                   Three forward passes. No extra parameters.

Data: Anthropic HH-RLHF dataset — real human preference pairs downloaded on first run.
Everything else in Claude (scale, filters, infrastructure) is efficiency on top of this.
"""

import os, gzip, json, math, random, argparse, urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--n_embd',     type=int,   default=64)
parser.add_argument('--n_layer',    type=int,   default=2)
parser.add_argument('--n_head',     type=int,   default=4)
parser.add_argument('--block_size', type=int,   default=128)
parser.add_argument('--sft_steps',  type=int,   default=500)
parser.add_argument('--dpo_steps',  type=int,   default=200)
parser.add_argument('--lr',         type=float, default=3e-3)
parser.add_argument('--beta',       type=float, default=0.1,  help='DPO KL penalty')
parser.add_argument('--max_pairs',  type=int,   default=2000, help='HH-RLHF pairs to load')
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()
random.seed(args.seed)
n_embd, block_size, n_layer, n_head = args.n_embd, args.block_size, args.n_layer, args.n_head
head_dim = n_embd // n_head

# ── DATA ──────────────────────────────────────────────────────────────────────
# Real Anthropic HH-RLHF: chosen/rejected conversation pairs.
# Each conversation is "\n\nHuman: ...\n\nAssistant: ..." multi-turn.
if not os.path.exists('hh_rlhf.jsonl.gz'):
    print("Downloading HH-RLHF dataset...")
    urllib.request.urlretrieve(
        'https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz',
        'hh_rlhf.jsonl.gz'
    )

pairs = []  # list of (chosen_str, rejected_str)
with gzip.open('hh_rlhf.jsonl.gz', 'rt') as f:
    for line in f:
        if len(pairs) >= args.max_pairs: break
        ex = json.loads(line)
        pairs.append((ex['chosen'], ex['rejected']))

random.shuffle(pairs)
print(f"loaded {len(pairs)} preference pairs")

# ── TOKENIZER ─────────────────────────────────────────────────────────────────
# Character-level. Role tags mark who is speaking — this is the structural
# difference between a language model and a chat assistant.
ROLE_TAGS = ['[H]', '[A]', '[BOS]', '[EOS]', '[PAD]']
all_text   = ''.join(c + r for c, r in pairs[:500])  # build vocab from data
chars      = sorted(set(all_text) - set(''.join(ROLE_TAGS))) + ROLE_TAGS
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
BOS, EOS, PAD = stoi['[BOS]'], stoi['[EOS]'], stoi['[PAD]']
H_TOK, A_TOK  = stoi['[H]'],   stoi['[A]']
print(f"vocab size: {vocab_size}")

def parse_turns(text):
    """Parse '\n\nHuman: ...\n\nAssistant: ...' into [(role, content), ...]"""
    turns = []
    for chunk in text.split('\n\nHuman: '):
        if not chunk: continue
        parts = chunk.split('\n\nAssistant: ', 1)
        turns.append(('H', parts[0].strip()))
        if len(parts) > 1:
            turns.append(('A', parts[1].strip()))
    return turns

def encode(text):
    """
    Encode a conversation into (token_ids, loss_mask).
    loss_mask[i]=1 only on assistant tokens — SFT trains on these positions only.
    This role-selective loss is the key structural difference from plain GPT.
    """
    turns = parse_turns(text)
    ids   = [BOS]; mask = [0]
    for role, content in turns:
        ids.append(A_TOK if role == 'A' else H_TOK); mask.append(0)
        for ch in content:
            ids.append(stoi.get(ch, PAD))
            mask.append(1 if role == 'A' else 0)
    ids.append(EOS); mask.append(1)
    return ids[:block_size], mask[:block_size]

# ── AUTOGRAD (micrograd by @karpathy) ─────────────────────────────────────────
class Value:
    def __init__(self, data, _ch=(), _op=''):
        self.data=float(data); self.grad=0.0
        self._back=lambda: None; self._prev=set(_ch)
    def __add__(self, o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data+o.data,(self,o))
        def _b(): self.grad+=out.grad; o.grad+=out.grad
        out._back=_b; return out
    def __mul__(self, o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data*o.data,(self,o))
        def _b(): self.grad+=o.data*out.grad; o.grad+=self.data*out.grad
        out._back=_b; return out
    def __pow__(self, e):
        out=Value(self.data**e,(self,))
        def _b(): self.grad+=e*self.data**(e-1)*out.grad
        out._back=_b; return out
    def log(self):
        x=max(self.data,1e-9); out=Value(math.log(x),(self,))
        def _b(): self.grad+=(1/x)*out.grad
        out._back=_b; return out
    def exp(self):
        x=math.exp(min(self.data,20)); out=Value(x,(self,))
        def _b(): self.grad+=x*out.grad
        out._back=_b; return out
    def relu(self):
        out=Value(max(0,self.data),(self,))
        def _b(): self.grad+=(self.data>0)*out.grad
        out._back=_b; return out
    def sigmoid(self):
        s=1/(1+math.exp(-max(-20,min(20,self.data)))); out=Value(s,(self,))
        def _b(): self.grad+=s*(1-s)*out.grad
        out._back=_b; return out
    def backward(self):
        topo=[]; seen=set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad=1.0
        for v in reversed(topo): v._back()
    def __neg__(self):        return self*-1
    def __radd__(self,o):     return self+o
    def __sub__(self,o):      return self+(-o if isinstance(o,Value) else Value(-o))
    def __rsub__(self,o):     return Value(o)+(-self)
    def __rmul__(self,o):     return self*o
    def __truediv__(self,o):  return self*o**-1
    def __rtruediv__(self,o): return Value(o)*self**-1

# ── MODEL ─────────────────────────────────────────────────────────────────────
rng = random.Random(args.seed)
mat = lambda r,c,s=0.02: [[Value(rng.gauss(0,s)) for _ in range(c)] for _ in range(r)]

def make_params():
    sd = {'wte': mat(vocab_size,n_embd), 'wpe': mat(block_size,n_embd)}
    for i in range(n_layer):
        sd[f'{i}.wq']=mat(n_embd,n_embd);   sd[f'{i}.wk']=mat(n_embd,n_embd)
        sd[f'{i}.wv']=mat(n_embd,n_embd);   sd[f'{i}.wo']=mat(n_embd,n_embd,0)
        sd[f'{i}.w1']=mat(4*n_embd,n_embd); sd[f'{i}.w2']=mat(n_embd,4*n_embd,0)
    return sd

θ = make_params()
all_params = [p for m in θ.values() for row in m for p in row]
print(f"num params: {len(all_params)}")

def mm(x,w):    return [sum(w[o][i]*x[i] for i in range(len(x))) for o in range(len(w))]
def softmax(lg):
    m=max(v.data for v in lg); ex=[(v-m).exp() for v in lg]; s=sum(ex); return [e/s for e in ex]
def rms(x):     s=sum(a*a for a in x)/len(x); return [a*(s+1e-5)**-0.5 for a in x]

def forward(tok, pos, kv):
    x=[t+p for t,p in zip(θ['wte'][tok], θ['wpe'][pos % block_size])]
    for i in range(n_layer):
        r=x; x=rms(x)
        q,k,v=mm(x,θ[f'{i}.wq']),mm(x,θ[f'{i}.wk']),mm(x,θ[f'{i}.wv'])
        kv[i][0].append(k); kv[i][1].append(v)
        out=[]
        for h in range(n_head):
            s=h*head_dim; qh=q[s:s+head_dim]
            K=[ki[s:s+head_dim] for ki in kv[i][0]]
            V=[vi[s:s+head_dim] for vi in kv[i][1]]
            sc=[sum(qh[j]*K[t][j] for j in range(head_dim))/head_dim**0.5 for t in range(len(K))]
            aw=softmax(sc)
            out.extend([sum(aw[t]*V[t][j] for t in range(len(V))) for j in range(head_dim)])
        x=[a+b for a,b in zip(mm(out,θ[f'{i}.wo']),r)]
        r=x; x=rms(x); x=[xi.relu()**2 for xi in mm(x,θ[f'{i}.w1'])]
        x=[a+b for a,b in zip(mm(x,θ[f'{i}.w2']),r)]
    return mm(x, θ['wte'])  # weight-tied lm head

def new_kv(): return [([],[]) for _ in range(n_layer)]

def seq_logprob(ids, mask, sd):
    """
    Full sequence log-prob, masked to assistant tokens only.
    Swaps param dict so we can evaluate both policy and reference model.
    """
    global θ; orig=θ; θ=sd
    kv=new_kv(); total=Value(0.0); n=0
    for t in range(len(ids)-1):
        logits=forward(ids[t], t, kv)
        if mask[t+1]: total=total+softmax(logits)[ids[t+1]].log(); n+=1
    θ=orig
    return total*(1/max(n,1))

# ── ADAM ──────────────────────────────────────────────────────────────────────
m_buf=[0.]*len(all_params); v_buf=[0.]*len(all_params)
def adam(step, lr):
    for j,p in enumerate(all_params):
        m_buf[j]=0.9*m_buf[j]+0.1*p.grad
        v_buf[j]=0.95*v_buf[j]+0.05*p.grad**2
        mh=m_buf[j]/(1-0.9**(step+1)); vh=v_buf[j]/(1-0.95**(step+1))
        p.data-=lr*mh/(vh**0.5+1e-8); p.grad=0.0

# ── PHASE 1: SFT ──────────────────────────────────────────────────────────────
# Train on chosen responses with assistant-only loss masking.
print("\n── SFT ──────────────────────────────────────────────────────────")
for step in range(args.sft_steps):
    chosen, _ = pairs[step % len(pairs)]
    ids, mask = encode(chosen)
    loss      = -seq_logprob(ids, mask, θ)
    loss.backward()
    adam(step, args.lr*(1-step/args.sft_steps))
    if (step+1) % 100 == 0:
        print(f"  {step+1:>5}/{args.sft_steps}  loss={loss.data:.4f}")

# ── PHASE 2: DPO ──────────────────────────────────────────────────────────────
# Freeze post-SFT weights as π_ref. Fine-tune θ on preference pairs.
#
# DPO loss = -log σ( β · [ log π_θ(yw|x) - log π_ref(yw|x)
#                          - log π_θ(yl|x) + log π_ref(yl|x) ] )
#
# No reward model. No PPO. Just this.
print("\n── DPO ──────────────────────────────────────────────────────────")
π_ref = {k: [[Value(p.data) for p in row] for row in m] for k,m in θ.items()}

for step in range(args.dpo_steps):
    chosen, rejected = pairs[step % len(pairs)]
    ids_w, mask_w    = encode(chosen)
    ids_l, mask_l    = encode(rejected)

    lp_w  = seq_logprob(ids_w, mask_w, θ)      # log π_θ(chosen)
    lp_l  = seq_logprob(ids_l, mask_l, θ)      # log π_θ(rejected)
    lp_rw = seq_logprob(ids_w, mask_w, π_ref)  # log π_ref(chosen)   — no grad
    lp_rl = seq_logprob(ids_l, mask_l, π_ref)  # log π_ref(rejected) — no grad

    margin = (lp_w - lp_rw.data) - (lp_l - lp_rl.data)  # implicit reward gap
    loss   = -(Value(args.beta) * margin).sigmoid().log()

    loss.backward()
    adam(args.sft_steps + step, args.lr * 0.1)

    if (step+1) % 50 == 0:
        print(f"  {step+1:>5}/{args.dpo_steps}  loss={loss.data:.4f}  margin={margin.data:.4f}")

# ── INFERENCE + CONSTITUTIONAL LOOP ──────────────────────────────────────────
# Generate → self-critique → revise. Same model, three forward passes.
def generate(prompt_str, max_new=80, temp=0.8):
    ids, _ = encode(prompt_str)
    kv     = new_kv(); logits = None
    for pos, tok in enumerate(ids):
        logits = forward(tok, pos, kv)
    out=[]; pos=len(ids)
    tok = random.choices(range(vocab_size), weights=[p.data for p in softmax(logits)])[0]
    while len(out) < max_new and tok not in (EOS, PAD):
        if tok not in (H_TOK, A_TOK): out.append(itos[tok])
        logits = forward(tok, pos, kv); pos += 1
        probs  = softmax(logits)
        if temp != 1.0:
            w=[p.data**(1/temp) for p in probs]; s=sum(w); w=[v/s for v in w]
            tok = random.choices(range(vocab_size), weights=w)[0]
        else:
            tok = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
    return ''.join(out)

def chat(human):
    """Generate draft → critique it → revise. Constitutional AI in 3 lines."""
    draft    = generate(f"\n\nHuman: {human}\n\nAssistant: ")
    critique = generate(f"\n\nHuman: Review this for helpfulness: \"{draft[:80]}\"\n\nAssistant: ")
    revised  = generate(f"\n\nHuman: {human}\n\nAssistant: {critique[:40]}\n\nHuman: go on\n\nAssistant: ")
    return draft, critique, revised

print("\n── generation ───────────────────────────────────────────────────")
for q in ["What can you help me with?", "Tell me something interesting.", "How do I stay safe online?"]:
    draft, crit, rev = chat(q)
    print(f"  human:    {q}")
    print(f"  draft:    {draft!r}")
    print(f"  critique: {crit!r}")
    print(f"  revised:  {rev!r}\n")
