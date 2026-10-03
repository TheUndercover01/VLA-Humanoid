"""Motion vocabulary: a skill-conditioned VAE over retargeted clips.

Each clip (data/processed/clips/<skill>_<idx>.npz, interface contract in CLOUD_PROMPT.md) is
resampled to N_WAYPOINTS waypoints over its duration and expressed relative to its first pose:
(dx, dy, dz, dyaw, grip), grip absolute. The decoder maps (skill, z) to such a motion.

    python -m motion.vae --clips data/processed/clips --out data/processed/vocab.pt
    python -m motion.vae --clips data/processed/clips_standin --out data/processed/vocab_standin.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SKILLS = ["reach", "push", "pick-lift", "place-down"]
N_WAYPOINTS = 10


class SkillDecoder(nn.Module):
    def __init__(self, n_skills, latent_dim, n_waypoints, hidden=128):
        super().__init__()
        self.n_skills, self.latent_dim, self.n_waypoints = n_skills, latent_dim, n_waypoints
        self.net = nn.Sequential(nn.Linear(n_skills + latent_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, n_waypoints * 5))
        # per-skill mean motion and scale, so the network only models the variation
        self.register_buffer("mean", torch.zeros(n_skills, n_waypoints, 5))
        self.register_buffer("scale", torch.ones(n_skills, 1, 5))

    def decode(self, skill_idx, z):
        """skill_idx (B,) long, z (B, latent_dim) -> (B, n_waypoints, 5): dx, dy, dz, dyaw, grip."""
        onehot = nn.functional.one_hot(skill_idx, self.n_skills).float()
        out = self.net(torch.cat([onehot, z], dim=-1)).view(-1, self.n_waypoints, 5)
        motion = self.mean[skill_idx] + out * self.scale[skill_idx]
        return torch.cat([motion[..., :4], motion[..., 4:].clamp(0, 1)], dim=-1)

    def config(self):
        return {"n_skills": self.n_skills, "latent_dim": self.latent_dim, "n_waypoints": self.n_waypoints,
                "hidden": self.net[0].out_features}


class Encoder(nn.Module):
    def __init__(self, n_skills, latent_dim, n_waypoints, hidden=128):
        super().__init__()
        self.n_skills = n_skills
        self.net = nn.Sequential(nn.Linear(n_skills + n_waypoints * 5, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2 * latent_dim))

    def forward(self, skill_idx, x):
        onehot = nn.functional.one_hot(skill_idx, self.n_skills).float()
        mu, logvar = self.net(torch.cat([onehot, x.flatten(1)], dim=-1)).chunk(2, dim=-1)
        return mu, logvar


def load_clip(path, n_waypoints=N_WAYPOINTS):
    """Resample a clip to n_waypoints and express it relative to its first pose."""
    c = np.load(path, allow_pickle=True)
    t = c["t"]
    tq = np.linspace(t[0], t[-1], n_waypoints)
    pos = np.stack([np.interp(tq, t, c["ee_pos"][:, k]) for k in range(3)], axis=-1) - c["ee_pos"][0]
    yaw = np.interp(tq, t, np.unwrap(c["ee_yaw"])) - c["ee_yaw"][0]
    grip = np.interp(tq, t, c["grip"])
    return str(c["skill"]), np.concatenate([pos, yaw[:, None], grip[:, None]], axis=-1).astype(np.float32), t[-1] - t[0]


def train(clips_dir, out, latent_dim=3, epochs=3000, beta=0.01, seed=0):
    torch.manual_seed(seed)
    data = [load_clip(p) for p in sorted(Path(clips_dir).glob("*.npz"))]
    skills = [s for s in SKILLS if any(d[0] == s for d in data)]
    idx = torch.tensor([skills.index(d[0]) for d in data])
    x = torch.tensor(np.stack([d[1] for d in data]))
    dt = float(np.median([d[2] for d in data])) / (N_WAYPOINTS - 1)

    dec = SkillDecoder(len(skills), latent_dim, N_WAYPOINTS)
    enc = Encoder(len(skills), latent_dim, N_WAYPOINTS)
    for k in range(len(skills)):
        xs = x[idx == k]
        dec.mean[k] = xs.mean(0)
        dec.scale[k, 0] = xs.std(0).mean(0).clamp(min=1e-3)
    xn = (x - dec.mean[idx]) / dec.scale[idx]                     # normalised targets per skill
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.net.parameters()), lr=1e-3)
    for ep in range(epochs):
        mu, logvar = enc(idx, xn)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = dec.net(torch.cat([nn.functional.one_hot(idx, len(skills)).float(), z], -1)).view_as(xn)
        rec_loss = ((rec - xn) ** 2).mean()
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
        loss = rec_loss + beta * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 500 == 0 or ep == epochs - 1:
            print(f"epoch {ep:5d}  recon {rec_loss.item():.4f}  kl {kl.item():.3f}")

    with torch.no_grad():
        mu, _ = enc(idx, xn)
        err = (dec.decode(idx, mu) - x)[..., :3].norm(dim=-1).mean(-1)   # mean waypoint position error
    for k, s in enumerate(skills):
        print(f"{s:10s} {int((idx == k).sum()):3d} clips, reconstruction error {err[idx == k].mean() * 1000:5.1f} mm")
    torch.save({"skills": skills, "latent_dim": latent_dim, "n_waypoints": N_WAYPOINTS, "dt": dt,
                "decoder_state_dict": dec.state_dict(), "decoder_config": dec.config(),
                "source": str(clips_dir)}, out)
    print(f"wrote {out} (dt {dt:.3f} s, {N_WAYPOINTS} waypoints = {dt * (N_WAYPOINTS - 1):.2f} s per motion)")


def load_vocab(path, device="cpu"):
    """Returns (decoder, vocab dict)."""
    v = torch.load(path, map_location=device, weights_only=False)
    dec = SkillDecoder(**v["decoder_config"]).to(device)
    dec.load_state_dict(v["decoder_state_dict"])
    return dec.eval(), v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--beta", type=float, default=0.01)
    a = ap.parse_args()
    train(a.clips, a.out, a.latent_dim, a.epochs, a.beta)
