"""Dataset and batching helpers for ligand-context surrogate affinity studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset


POCKET_NODE = 0
LIGAND_NODE = 1

EDGE_LIGAND_LIGAND = 0
EDGE_POCKET_TO_LIGAND = 1

_RESULT_KEYS = ("all_results", "results")
_POSITION_KEYS = ("pred_pos", "ligand_pos", "pos")
_ATOMIC_NUMBER_KEYS = ("ligand_atomic_numbers", "ligand_z", "z_lig", "atomic_numbers", "z")
_AFFINITY_KEYS = ("affinity", "y", "target")

_ELEMENT_TO_ATOMIC_NUMBER = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "BR": 35,
    "I": 53,
}


@dataclass(frozen=True, slots=True)
class SurrogateGraph:
    pos: torch.Tensor
    z: torch.Tensor
    node_type: torch.Tensor
    y: torch.Tensor
    ligand_size: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class SurrogateBatch:
    pos: torch.Tensor
    z: torch.Tensor
    node_type: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    ligand_sizes: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    t: torch.Tensor | None = None
    clean_pos: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        if self.y.ndim == 0:
            return 1
        return int(self.y.shape[0])

    def to(self, device: torch.device | str) -> "SurrogateBatch":
        self.pos = self.pos.to(device)
        self.z = self.z.to(device)
        self.node_type = self.node_type.to(device)
        self.batch = self.batch.to(device)
        self.y = self.y.to(device)
        self.ligand_sizes = self.ligand_sizes.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        if self.t is not None:
            self.t = self.t.to(device)
        if self.clean_pos is not None:
            self.clean_pos = self.clean_pos.to(device)
        return self

    def clone(self) -> "SurrogateBatch":
        return SurrogateBatch(
            pos=self.pos.clone(),
            z=self.z.clone(),
            node_type=self.node_type.clone(),
            batch=self.batch.clone(),
            y=self.y.clone(),
            ligand_sizes=self.ligand_sizes.clone(),
            edge_index=self.edge_index.clone(),
            edge_type=self.edge_type.clone(),
            t=None if self.t is None else self.t.clone(),
            clean_pos=None if self.clean_pos is None else self.clean_pos.clone(),
        )


class LigandContextSurrogateDataset(Dataset[SurrogateGraph]):
    """Load ligand+pocket affinity records from one or more `.pt` files."""

    def __init__(
        self,
        *,
        pt_path: str | Path | None = None,
        pt_dir: str | Path | None = None,
        pdb_path: str | Path,
        vina_mode: str = "score_only",
        max_affinity: float | None = None,
        results_key: str | None = None,
    ) -> None:
        if pt_path is None and pt_dir is None:
            raise ValueError("Provide data.pt_path or data.pt_dir")
        if pt_path is not None and pt_dir is not None:
            raise ValueError("Provide only one of data.pt_path or data.pt_dir")
        if vina_mode not in {"dock", "minimize", "score_only"}:
            raise ValueError(f"Unsupported vina_mode {vina_mode!r}")

        self.source_path = Path(pt_path) if pt_path is not None else Path(pt_dir)
        self.vina_mode = vina_mode
        self.max_affinity = max_affinity
        self.results_key = results_key
        self.pocket_pos, self.pocket_z = load_pocket_atomic_numbers(Path(pdb_path))

        results = _load_surrogate_results(
            pt_path=None if pt_path is None else Path(pt_path),
            pt_dir=None if pt_dir is None else Path(pt_dir),
            results_key=results_key,
        )
        self.records = []
        for index, record in enumerate(results):
            affinity = _extract_affinity(record, vina_mode=vina_mode)
            if max_affinity is not None and affinity > float(max_affinity):
                continue
            self.records.append({"index": index, "record": record, "affinity": affinity})
        if not self.records:
            raise ValueError("No surrogate records available after filtering")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> SurrogateGraph:
        entry = self.records[idx]
        record = entry["record"]
        ligand_pos = _extract_ligand_positions(record)
        ligand_z = _extract_ligand_atomic_numbers(record, expected_num_atoms=ligand_pos.shape[0])

        pos = torch.cat([ligand_pos, self.pocket_pos], dim=0)
        z = torch.cat([ligand_z, self.pocket_z], dim=0)
        node_type = torch.cat(
            [
                torch.full((ligand_pos.shape[0],), LIGAND_NODE, dtype=torch.long),
                torch.full((self.pocket_pos.shape[0],), POCKET_NODE, dtype=torch.long),
            ],
            dim=0,
        )
        metadata = {
            "record_index": entry["index"],
            "smiles": record.get("smiles"),
        }
        return SurrogateGraph(
            pos=pos,
            z=z,
            node_type=node_type,
            y=torch.tensor(float(entry["affinity"]), dtype=torch.float32),
            ligand_size=int(ligand_pos.shape[0]),
            metadata=metadata,
        )


class SurrogateNoiseSchedule:
    """Diffusion-style Gaussian corruption for ligand coordinates."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError("beta_start and beta_end must be > 0")
        if beta_end < beta_start:
            raise ValueError("beta_end must be >= beta_start")
        self.enabled = bool(enabled)
        self.n_steps = int(n_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        betas = torch.linspace(self.beta_start, self.beta_end, self.n_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def sample_batch(
        self,
        batch: SurrogateBatch,
        *,
        include_timestep: bool,
        r_ligand: float,
        r_cross: float,
        generator: torch.Generator | None = None,
    ) -> SurrogateBatch:
        prepared = batch.clone()
        prepared.clean_pos = batch.pos.clone()

        if not self.enabled:
            timestep = torch.zeros((prepared.batch_size,), dtype=torch.long, device=prepared.pos.device)
            prepared.t = timestep if include_timestep else None
            prepared.edge_index, prepared.edge_type = build_context_edges(
                prepared.pos,
                prepared.node_type,
                batch=prepared.batch,
                r_ligand=r_ligand,
                r_cross=r_cross,
            )
            return prepared

        timestep = torch.randint(
            low=0,
            high=self.n_steps,
            size=(prepared.batch_size,),
            generator=generator,
            device=prepared.pos.device,
            dtype=torch.long,
        )
        alpha_bar = self.alpha_bars.to(prepared.pos.device).index_select(0, timestep)
        node_alpha = alpha_bar.index_select(0, prepared.batch).unsqueeze(-1)

        ligand_mask = prepared.node_type == LIGAND_NODE
        noise = torch.randn(
            prepared.pos.shape,
            generator=generator,
            device=prepared.pos.device,
            dtype=prepared.pos.dtype,
        )
        noisy_pos = prepared.pos.clone()
        noisy_pos[ligand_mask] = (
            node_alpha[ligand_mask].sqrt() * prepared.pos[ligand_mask]
            + (1.0 - node_alpha[ligand_mask]).sqrt() * noise[ligand_mask]
        )
        prepared.pos = noisy_pos
        prepared.t = timestep if include_timestep else None
        prepared.edge_index, prepared.edge_type = build_context_edges(
            prepared.pos,
            prepared.node_type,
            batch=prepared.batch,
            r_ligand=r_ligand,
            r_cross=r_cross,
        )
        return prepared


def load_pocket_atomic_numbers(pdb_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    coords: list[list[float]] = []
    atomic_numbers: list[int] = []
    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip() or line[12:16].strip()[:2].strip()
            coords.append([x, y, z])
            atomic_numbers.append(_atomic_number_from_symbol(elem))
    if not coords:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path}")
    return (
        torch.tensor(coords, dtype=torch.float32),
        torch.tensor(atomic_numbers, dtype=torch.long),
    )


def build_context_edges(
    pos: torch.Tensor,
    node_type: torch.Tensor,
    *,
    batch: torch.Tensor | None = None,
    r_ligand: float = 5.0,
    r_cross: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (num_nodes, 3), got {tuple(pos.shape)}")
    if node_type.ndim != 1 or node_type.shape[0] != pos.shape[0]:
        raise ValueError("node_type must have shape (num_nodes,)")
    if batch is None:
        batch = torch.zeros((pos.shape[0],), dtype=torch.long, device=pos.device)
    if batch.ndim != 1 or batch.shape[0] != pos.shape[0]:
        raise ValueError("batch must have shape (num_nodes,)")

    edge_src: list[torch.Tensor] = []
    edge_dst: list[torch.Tensor] = []
    edge_type: list[torch.Tensor] = []

    for graph_id in batch.unique(sorted=True):
        graph_mask = batch == graph_id
        graph_indices = graph_mask.nonzero(as_tuple=True)[0]
        if graph_indices.numel() == 0:
            continue
        sub_pos = pos.index_select(0, graph_indices)
        sub_type = node_type.index_select(0, graph_indices)

        ligand_local = (sub_type == LIGAND_NODE).nonzero(as_tuple=True)[0]
        pocket_local = (sub_type == POCKET_NODE).nonzero(as_tuple=True)[0]

        if ligand_local.numel() > 0:
            ligand_pos = sub_pos.index_select(0, ligand_local)
            dist_ll = torch.cdist(ligand_pos, ligand_pos, p=2)
            keep_ll = dist_ll <= float(r_ligand)
            keep_ll = keep_ll & ~torch.eye(ligand_local.numel(), dtype=torch.bool, device=pos.device)
            src_ll, dst_ll = keep_ll.nonzero(as_tuple=True)
            if src_ll.numel() > 0:
                edge_src.append(graph_indices.index_select(0, ligand_local.index_select(0, src_ll)))
                edge_dst.append(graph_indices.index_select(0, ligand_local.index_select(0, dst_ll)))
                edge_type.append(
                    torch.full((src_ll.numel(),), EDGE_LIGAND_LIGAND, dtype=torch.long, device=pos.device)
                )

        if ligand_local.numel() > 0 and pocket_local.numel() > 0:
            ligand_pos = sub_pos.index_select(0, ligand_local)
            pocket_pos = sub_pos.index_select(0, pocket_local)
            dist_pl = torch.cdist(pocket_pos, ligand_pos, p=2)
            src_pl, dst_pl = (dist_pl <= float(r_cross)).nonzero(as_tuple=True)
            if src_pl.numel() > 0:
                edge_src.append(graph_indices.index_select(0, pocket_local.index_select(0, src_pl)))
                edge_dst.append(graph_indices.index_select(0, ligand_local.index_select(0, dst_pl)))
                edge_type.append(
                    torch.full((src_pl.numel(),), EDGE_POCKET_TO_LIGAND, dtype=torch.long, device=pos.device)
                )

    if not edge_src:
        return (
            torch.empty((2, 0), dtype=torch.long, device=pos.device),
            torch.empty((0,), dtype=torch.long, device=pos.device),
        )
    return (
        torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0),
        torch.cat(edge_type, dim=0),
    )


def collate_surrogate_graphs(
    graphs: Sequence[SurrogateGraph],
    *,
    r_ligand: float,
    r_cross: float,
) -> SurrogateBatch:
    if not graphs:
        raise ValueError("Cannot collate an empty batch")
    pos_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    node_type_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    ligand_sizes: list[int] = []
    targets: list[torch.Tensor] = []

    for graph_index, graph in enumerate(graphs):
        pos_parts.append(graph.pos)
        z_parts.append(graph.z)
        node_type_parts.append(graph.node_type)
        batch_parts.append(torch.full((graph.pos.shape[0],), graph_index, dtype=torch.long))
        ligand_sizes.append(int(graph.ligand_size))
        targets.append(graph.y.reshape(()))

    pos = torch.cat(pos_parts, dim=0)
    z = torch.cat(z_parts, dim=0)
    node_type = torch.cat(node_type_parts, dim=0)
    batch = torch.cat(batch_parts, dim=0)
    edge_index, edge_type = build_context_edges(
        pos,
        node_type,
        batch=batch,
        r_ligand=r_ligand,
        r_cross=r_cross,
    )
    return SurrogateBatch(
        pos=pos,
        z=z,
        node_type=node_type,
        batch=batch,
        y=torch.stack(targets, dim=0),
        ligand_sizes=torch.tensor(ligand_sizes, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        clean_pos=pos.clone(),
    )


def _load_surrogate_results(
    *,
    pt_path: Path | None,
    pt_dir: Path | None,
    results_key: str | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pt_paths = [pt_path] if pt_path is not None else sorted(Path(pt_dir).glob("*.pt"))
    if not pt_paths:
        raise FileNotFoundError(f"No .pt files found under {pt_dir}")
    for path in pt_paths:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Loading {path} requires an optional dependency ({exc.name}). "
                "If the records store RDKit molecules, install RDKit in this environment "
                "or export the records with explicit ligand_atomic_numbers instead."
            ) from exc
        if isinstance(loaded, list):
            chunk = loaded
        elif isinstance(loaded, dict):
            chunk = None
            if results_key is not None and results_key in loaded:
                chunk = loaded[results_key]
            if chunk is None:
                for candidate in _RESULT_KEYS:
                    if candidate in loaded:
                        chunk = loaded[candidate]
                        break
            if chunk is None:
                raise KeyError(
                    f"Could not find result list in {path}; looked for {results_key!r} and {_RESULT_KEYS}"
                )
        else:
            raise TypeError(f"Unsupported record payload type {type(loaded)!r} in {path}")
        results.extend([dict(item) for item in chunk])
    return results


def _extract_ligand_positions(record: dict[str, Any]) -> torch.Tensor:
    for key in _POSITION_KEYS:
        value = record.get(key)
        if value is not None:
            pos = torch.as_tensor(value, dtype=torch.float32)
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"{key} must have shape (num_atoms, 3), got {tuple(pos.shape)}")
            return pos
    raise KeyError(f"Record is missing ligand positions; looked for {_POSITION_KEYS}")


def _extract_ligand_atomic_numbers(record: dict[str, Any], *, expected_num_atoms: int) -> torch.Tensor:
    for key in _ATOMIC_NUMBER_KEYS:
        value = record.get(key)
        if value is not None:
            z = torch.as_tensor(value, dtype=torch.long).reshape(-1)
            if z.shape[0] != expected_num_atoms:
                raise ValueError(
                    f"{key} has {z.shape[0]} atoms but positions have {expected_num_atoms}"
                )
            return z

    mol = record.get("mol")
    if mol is not None:
        try:
            import rdkit  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This record stores an RDKit molecule. Install rdkit or export ligand_atomic_numbers in the .pt file."
            ) from exc
        z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        if z.shape[0] != expected_num_atoms:
            raise ValueError(
                f"mol has {z.shape[0]} atoms but positions have {expected_num_atoms}"
            )
        return z

    raise KeyError(f"Record is missing ligand atomic numbers; looked for {_ATOMIC_NUMBER_KEYS} and mol")


def _extract_affinity(record: dict[str, Any], *, vina_mode: str) -> float:
    for key in _AFFINITY_KEYS:
        value = record.get(key)
        if value is not None:
            tensor_value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            if tensor_value.numel() == 0:
                continue
            return float(tensor_value[0].item())

    vina = record.get("vina")
    if isinstance(vina, dict):
        vina_values = vina.get(vina_mode)
        if isinstance(vina_values, list) and vina_values:
            first = vina_values[0]
            if isinstance(first, dict) and "affinity" in first:
                return float(first["affinity"])
    raise KeyError(
        f"Record is missing an affinity target; looked for {_AFFINITY_KEYS} or vina[{vina_mode!r}][0]['affinity']"
    )


def _atomic_number_from_symbol(symbol: str) -> int:
    return int(_ELEMENT_TO_ATOMIC_NUMBER.get(symbol.strip().upper(), 0))
