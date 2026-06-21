"""
Transformer-based Inverse Kinematics — Minimal Example
-------------------------------------------------------
Flow:
  1. Synthetic data: sample random joint angles → FK → target pose
  2. Robot encoder: joint tokens (DH params) → robot context via transformer encoder
  3. IK decoder: [pose_token + joint tokens] → predicted joint angles
  4. Loss: MSE on angles + FK consistency (reprojection) loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ---------------------------------------------------------------------------
# 1. Forward Kinematics (simplified planar-safe, works for any serial chain)
# ---------------------------------------------------------------------------

def dh_matrix(a, alpha, d, theta):
    """Standard DH transformation matrix. All inputs are tensors."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    T = torch.stack([
        torch.stack([ct,    -st*ca,  st*sa,  a*ct]),
        torch.stack([st,     ct*ca, -ct*sa,  a*st]),
        torch.stack([torch.zeros_like(ct), sa, ca, d]),
        torch.stack([torch.zeros_like(ct),
                     torch.zeros_like(ct),
                     torch.zeros_like(ct),
                     torch.ones_like(ct)]),
    ])  # (4, 4)
    return T


def forward_kinematics(joint_angles, dh_params):
    """
    Compute FK for a serial chain.

    Args:
        joint_angles: (n_joints,) tensor of joint angles in radians
        dh_params:    (n_joints, 4) tensor — columns: [a, alpha, d, theta_offset]

    Returns:
        position:    (3,) end-effector xyz
        rotation:    (3, 3) end-effector rotation matrix
    """
    T = torch.eye(4)
    for i in range(len(joint_angles)):
        a, alpha, d, offset = dh_params[i]
        theta = joint_angles[i] + offset
        T = T @ dh_matrix(a, alpha, d, theta)
    return T[:3, 3], T[:3, :3]


def rotation_to_6d(R):
    """Convert (3,3) rotation matrix to continuous 6D representation (first 2 columns)."""
    return R[:, :2].reshape(6)   # (6,)


def pose_to_vector(position, rotation):
    """Concatenate position + 6D rotation into a 9-dim pose vector."""
    return torch.cat([position, rotation_to_6d(rotation)])  # (9,)


# ---------------------------------------------------------------------------
# 2. Dataset — generates synthetic (robot_params, pose, joint_angles) tuples
# ---------------------------------------------------------------------------

class IKDataset(Dataset):
    """
    Generates random robots and random joint configurations.
    Each sample:
        dh_params:     (n_joints, 7) — [a, alpha, d, offset, type, lo, hi]
        pose:          (9,)          — [position(3) + 6D rotation(6)]
        joint_angles:  (n_joints,)   — ground truth angles (the IK solution)
    """

    def __init__(self, n_samples=50_000, n_joints=6, randomize_robot=False):
        self.n_samples = n_samples
        self.n_joints = n_joints
        self.randomize_robot = randomize_robot

        # If not randomizing, fix one robot (UR5-like proportions)
        if not randomize_robot:
            self.fixed_dh = self._sample_robot_dh(n_joints)

    def _sample_robot_dh(self, n_joints):
        """Sample plausible DH parameters for a random serial robot."""
        a      = torch.FloatTensor(n_joints).uniform_(0.0, 0.5)   # link lengths (m)
        alpha  = torch.FloatTensor(n_joints).uniform_(-np.pi/2, np.pi/2)
        d      = torch.FloatTensor(n_joints).uniform_(0.0, 0.2)
        offset = torch.zeros(n_joints)
        lo     = torch.full((n_joints,), -np.pi)
        hi     = torch.full((n_joints,),  np.pi)
        # Stack into (n_joints, 6): [a, alpha, d, offset, lo, hi]
        return torch.stack([a, alpha, d, offset, lo, hi], dim=1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        dh = self.fixed_dh if not self.randomize_robot else self._sample_robot_dh(self.n_joints)

        # Sample a random valid joint configuration
        lo, hi = dh[:, 4], dh[:, 5]
        angles = lo + (hi - lo) * torch.rand(self.n_joints)

        # FK to get the pose this configuration produces
        pos, rot = forward_kinematics(angles, dh[:, :4])
        pose = pose_to_vector(pos, rot)

        return {
            "dh_params":    dh,        # (n_joints, 6)
            "pose":         pose,      # (9,)
            "joint_angles": angles,    # (n_joints,)
        }


# ---------------------------------------------------------------------------
# 3. Model — Transformer Encoder-Decoder
# ---------------------------------------------------------------------------

JOINT_TOKEN_DIM = 6     # [a, alpha, d, offset, lo, hi] — raw DH features
POSE_DIM        = 9     # position(3) + 6D rotation(6)
D_MODEL         = 128   # transformer hidden size


class JointEmbedding(nn.Module):
    """Projects raw per-joint DH features into d_model space."""
    def __init__(self, joint_dim=JOINT_TOKEN_DIM, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(joint_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, dh_params):
        """dh_params: (B, n_joints, joint_dim) → (B, n_joints, d_model)"""
        return self.proj(dh_params)


class IKTransformer(nn.Module):
    """
    Encoder: joint tokens (robot description) → robot context
    Decoder: pose token + robot context → joint angle predictions

    Encoder input:  (B, n_joints, d_model)  — one token per joint
    Decoder input:  (B, 1, d_model)         — pose token (query)
    Decoder output: (B, n_joints, 1)        — one angle per joint
    """

    def __init__(
        self,
        d_model=D_MODEL,
        n_heads=4,
        n_encoder_layers=3,
        n_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_joints=8,
    ):
        super().__init__()

        # --- Embeddings ---
        self.joint_embedding = JointEmbedding(JOINT_TOKEN_DIM, d_model)
        self.pose_proj = nn.Linear(POSE_DIM, d_model)

        # Learned positional encoding for joints (chain position awareness)
        self.pos_embedding = nn.Embedding(max_joints, d_model)

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # --- Output head: predict one angle per joint ---
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, dh_params, pose):
        """
        Args:
            dh_params: (B, n_joints, 6)  — per-joint DH features
            pose:      (B, 9)            — target end-effector pose

        Returns:
            joint_angles: (B, n_joints)  — predicted joint angles
        """
        B, n_joints, _ = dh_params.shape

        # --- Encode robot structure ---
        joint_tokens = self.joint_embedding(dh_params)                  # (B, n_joints, d_model)
        positions = torch.arange(n_joints, device=dh_params.device)
        joint_tokens = joint_tokens + self.pos_embedding(positions)     # add positional info
        memory = self.encoder(joint_tokens)                              # (B, n_joints, d_model)

        # --- Decode: pose token queries the robot context ---
        pose_token = self.pose_proj(pose).unsqueeze(1)                  # (B, 1, d_model)

        # Expand pose token to one query per joint so decoder outputs n_joints tokens
        # Each query is the pose token + positional embedding for that joint index
        queries = pose_token.expand(-1, n_joints, -1) + self.pos_embedding(positions)
        decoded = self.decoder(queries, memory)                         # (B, n_joints, d_model)

        # --- Predict angle per joint ---
        angles = self.output_head(decoded).squeeze(-1)                  # (B, n_joints)
        return angles


# ---------------------------------------------------------------------------
# 4. FK Consistency Loss — differentiable reprojection
# ---------------------------------------------------------------------------

def fk_consistency_loss(predicted_angles, dh_params, target_pose):
    """
    Run FK on predicted joint angles and measure distance to target pose.
    This is the key loss that grounds predictions in physical reality.

    Args:
        predicted_angles: (B, n_joints)
        dh_params:        (B, n_joints, 6)
        target_pose:      (B, 9)

    Returns:
        scalar loss
    """
    B = predicted_angles.shape[0]
    total_loss = torch.tensor(0.0, requires_grad=True)

    for b in range(B):
        pos, rot = forward_kinematics(predicted_angles[b], dh_params[b, :, :4])
        pred_pose = pose_to_vector(pos, rot)
        total_loss = total_loss + nn.functional.mse_loss(pred_pose, target_pose[b])

    return total_loss / B


# ---------------------------------------------------------------------------
# 5. Training Loop
# ---------------------------------------------------------------------------

def train(
    n_joints=6,
    n_samples=20_000,
    batch_size=64,
    n_epochs=20,
    lr=1e-3,
    fk_loss_weight=0.5,    # weight for FK consistency vs direct angle MSE
    device="cpu",
):
    print(f"Training on {device} | {n_joints} DOF | {n_samples} samples")

    dataset    = IKDataset(n_samples=n_samples, n_joints=n_joints)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model     = IKTransformer(max_joints=n_joints).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        angle_loss_sum = 0.0
        fk_loss_sum = 0.0

        for batch in dataloader:
            dh_params    = batch["dh_params"].to(device)     # (B, n_joints, 6)
            pose         = batch["pose"].to(device)           # (B, 9)
            target_angles = batch["joint_angles"].to(device)  # (B, n_joints)

            optimizer.zero_grad()

            # Forward pass
            pred_angles = model(dh_params, pose)              # (B, n_joints)

            # Loss 1: direct supervision on joint angles
            angle_loss = nn.functional.mse_loss(pred_angles, target_angles)

            # Loss 2: FK consistency (does the predicted config actually reach the pose?)
            fk_loss = fk_consistency_loss(pred_angles, dh_params, pose)

            loss = angle_loss + fk_loss_weight * fk_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += loss.item()
            angle_loss_sum += angle_loss.item()
            fk_loss_sum   += fk_loss.item()

        scheduler.step()
        n_batches = len(dataloader)
        print(
            f"Epoch {epoch+1:>3}/{n_epochs} | "
            f"Loss: {total_loss/n_batches:.4f} | "
            f"Angle: {angle_loss_sum/n_batches:.4f} | "
            f"FK: {fk_loss_sum/n_batches:.4f}"
        )

    return model


# ---------------------------------------------------------------------------
# 6. Inference example
# ---------------------------------------------------------------------------

def predict_ik(model, dh_params, target_pose, device="cpu"):
    """
    Run IK inference for a single robot + target pose.

    Args:
        dh_params:   (n_joints, 6) tensor
        target_pose: (9,) tensor — [position(3) + 6D rotation(6)]

    Returns:
        predicted joint angles (n_joints,)
    """
    model.eval()
    with torch.no_grad():
        angles = model(
            dh_params.unsqueeze(0).to(device),
            target_pose.unsqueeze(0).to(device),
        )
    return angles.squeeze(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train
    model = train(
        n_joints=6,
        n_samples=20_000,
        batch_size=64,
        n_epochs=20,
        lr=1e-3,
        fk_loss_weight=0.5,
        device=device,
    )

    # Quick inference test — generate a random sample and run IK on it
    dataset = IKDataset(n_samples=1, n_joints=6)
    sample  = dataset[0]

    pred_angles = predict_ik(model, sample["dh_params"], sample["pose"], device)

    print("\n--- Inference Test ---")
    print(f"Target angles (GT):  {sample['joint_angles'].numpy().round(3)}")
    print(f"Predicted angles:    {pred_angles.cpu().numpy().round(3)}")

    # Verify by running FK on prediction
    pred_pos, pred_rot = forward_kinematics(pred_angles.cpu(), sample["dh_params"][:, :4])
    tgt_pos  = sample["pose"][:3]
    pos_err  = (pred_pos - tgt_pos).norm().item()
    print(f"Position error:      {pos_err*1000:.2f} mm")