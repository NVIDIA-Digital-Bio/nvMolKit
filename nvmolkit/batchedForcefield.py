"""Python API for batched forcefield energy and gradient evaluation.

.. note::
   Calling :meth:`~MMFFBatchedForcefield.compute_energy` or
   :meth:`~MMFFBatchedForcefield.compute_gradients` individually from Python
   incurs per-call overhead that dominates the GPU computation time.
   Acceleration over RDKit should only be expected when these evaluations run
   inside the minimizer, where the native loop avoids repeated Python
   round-trips.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nvmolkit import _batchedForcefield  # type: ignore
from nvmolkit._mmff_bridge import default_rdkit_mmff_properties, make_internal_mmff_properties

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties


@dataclass
class _DistanceConstraint:
    idx1: int
    idx2: int
    relative: bool
    min_len: float
    max_len: float
    force_constant: float


@dataclass
class _PositionConstraint:
    idx: int
    max_displ: float
    force_constant: float


@dataclass
class _AngleConstraint:
    idx1: int
    idx2: int
    idx3: int
    relative: bool
    min_angle_deg: float
    max_angle_deg: float
    force_constant: float


@dataclass
class _TorsionConstraint:
    idx1: int
    idx2: int
    idx3: int
    idx4: int
    relative: bool
    min_dihedral_deg: float
    max_dihedral_deg: float
    force_constant: float


class MMFFBatchElement:
    """Mutable view over a single molecule inside an MMFF batch."""

    def __init__(self, parent: "MMFFBatchedForcefield", idx: int):
        self._parent = parent
        self._idx = idx

    @property
    def num_atoms(self) -> int:
        """Return the number of atoms in this batch element."""
        return self._parent._molecules[self._idx].GetNumAtoms()

    def add_distance_constraint(
        self,
        idx1: int,
        idx2: int,
        relative: bool,
        min_len: float,
        max_len: float,
        force_constant: float,
    ) -> None:
        """Add a distance constraint for this molecule."""
        self._parent._validate_atom_indices(self._idx, idx1, idx2)
        self._parent._distance_constraints[self._idx].append(
            _DistanceConstraint(idx1, idx2, relative, min_len, max_len, force_constant)
        )
        self._parent._dirty = True

    def add_position_constraint(
        self,
        idx: int,
        max_displ: float,
        force_constant: float,
    ) -> None:
        """Add a position constraint for this molecule."""
        self._parent._validate_atom_indices(self._idx, idx)
        self._parent._position_constraints[self._idx].append(
            _PositionConstraint(idx, max_displ, force_constant)
        )
        self._parent._dirty = True

    def add_angle_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        relative: bool,
        min_angle_deg: float,
        max_angle_deg: float,
        force_constant: float,
    ) -> None:
        """Add an angle constraint for this molecule."""
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3)
        self._parent._angle_constraints[self._idx].append(
            _AngleConstraint(idx1, idx2, idx3, relative, min_angle_deg, max_angle_deg, force_constant)
        )
        self._parent._dirty = True

    def add_torsion_constraint(
        self,
        idx1: int,
        idx2: int,
        idx3: int,
        idx4: int,
        relative: bool,
        min_dihedral_deg: float,
        max_dihedral_deg: float,
        force_constant: float,
    ) -> None:
        """Add a torsion constraint for this molecule."""
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3, idx4)
        self._parent._torsion_constraints[self._idx].append(
            _TorsionConstraint(
                idx1, idx2, idx3, idx4, relative, min_dihedral_deg, max_dihedral_deg, force_constant
            )
        )
        self._parent._dirty = True


class MMFFBatchedForcefield:
    """Evaluate MMFF energies and gradients for a batch of molecules."""

    def __init__(
        self,
        molecules: list["Mol"],
        properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None" = None,
        conf_id: int | Sequence[int] = -1,
        nonBondedThreshold: float | Sequence[float] = 100.0,
        ignoreInterfragInteractions: bool | Sequence[bool] = True,
    ):
        """Create a batched MMFF forcefield wrapper.

        Args:
            molecules: Molecules to evaluate.
            properties: RDKit ``MMFFMolProperties`` object, a per-molecule list
                of those objects, or ``None`` to use default MMFF94 settings.
            conf_id: Conformer id or per-molecule conformer ids to use.
            nonBondedThreshold: Non-bonded cutoff distance or per-molecule values.
            ignoreInterfragInteractions: Whether to omit interfragment non-bonded
                interactions, as a scalar or per-molecule list.
        """
        self._molecules = molecules
        self._properties = self._normalize_properties(properties)
        self._conf_ids = self._normalize_conf_ids(conf_id)
        self._non_bonded_thresholds = self._normalize_scalar_or_list(
            nonBondedThreshold, "nonBondedThreshold"
        )
        self._ignore_interfrag_interactions = self._normalize_scalar_or_list(
            ignoreInterfragInteractions, "ignoreInterfragInteractions"
        )
        self._distance_constraints: list[list[_DistanceConstraint]] = [[] for _ in molecules]
        self._position_constraints: list[list[_PositionConstraint]] = [[] for _ in molecules]
        self._angle_constraints: list[list[_AngleConstraint]] = [[] for _ in molecules]
        self._torsion_constraints: list[list[_TorsionConstraint]] = [[] for _ in molecules]
        self._native_ff = None
        self._dirty = True
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def __len__(self) -> int:
        """Return the number of molecules in the batch."""
        return len(self._molecules)

    def __getitem__(self, idx: int) -> MMFFBatchElement:
        """Return a mutable view for one molecule in the batch."""
        if idx < 0 or idx >= len(self._molecules):
            raise IndexError(f"Batch element index {idx} out of range")
        return MMFFBatchElement(self, idx)

    def _normalize_properties(
        self, properties: "RDKitMMFFMolProperties | Sequence[RDKitMMFFMolProperties | None] | None"
    ) -> list["RDKitMMFFMolProperties | None"]:
        if properties is None:
            return [None for _ in self._molecules]
        if isinstance(properties, Sequence) and not hasattr(properties, "SetMMFFVariant"):
            if len(properties) != len(self._molecules):
                raise ValueError(
                    f"Expected {len(self._molecules)} MMFFMolProperties objects, got {len(properties)}"
                )
            return list(properties)
        return [properties for _ in self._molecules]

    def _normalize_scalar_or_list(self, value, name: str):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != len(self._molecules):
                raise ValueError(f"Expected {len(self._molecules)} values for {name}, got {len(value)}")
            return list(value)
        return [value for _ in self._molecules]

    def _default_mmff_properties(self, mol: "Mol"):
        return default_rdkit_mmff_properties(mol)

    def _copy_mmff_properties(self, mol: "Mol", properties, non_bonded_threshold: float, ignore_interfrag_interactions: bool):
        """Convert public RDKit MMFF properties into nvMolKit's internal transport."""
        source = self._default_mmff_properties(mol) if properties is None else properties
        return make_internal_mmff_properties(
            source,
            non_bonded_threshold=float(non_bonded_threshold),
            ignore_interfrag_interactions=bool(ignore_interfrag_interactions),
        )

    def _normalize_conf_ids(self, conf_id: int | Sequence[int]) -> list[int]:
        if isinstance(conf_id, int):
            return [conf_id for _ in self._molecules]
        if len(conf_id) != len(self._molecules):
            raise ValueError(f"Expected {len(self._molecules)} conf_id values, got {len(conf_id)}")
        return list(conf_id)

    def _validate_atom_indices(self, batch_idx: int, *indices: int) -> None:
        num_atoms = self._molecules[batch_idx].GetNumAtoms()
        for idx in indices:
            if idx < 0 or idx >= num_atoms:
                raise IndexError(
                    f"Atom index {idx} out of range for molecule {batch_idx} with {num_atoms} atoms"
                )

    def _build(self) -> None:
        """Rebuild the native batched forcefield from the current state."""
        if not self._molecules:
            self._native_ff = None
            self._dirty = False
            return
        native_properties = [
            self._copy_mmff_properties(mol, props, threshold, ignore)
            for mol, props, threshold, ignore in zip(
                self._molecules,
                self._properties,
                self._non_bonded_thresholds,
                self._ignore_interfrag_interactions,
            )
        ]
        distance_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.relative,
                    c.min_len,
                    c.max_len,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._distance_constraints
        ]
        position_constraints = [
            [(c.idx, c.max_displ, c.force_constant) for c in constraints]
            for constraints in self._position_constraints
        ]
        angle_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.idx3,
                    c.relative,
                    c.min_angle_deg,
                    c.max_angle_deg,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._angle_constraints
        ]
        torsion_constraints = [
            [
                (
                    c.idx1,
                    c.idx2,
                    c.idx3,
                    c.idx4,
                    c.relative,
                    c.min_dihedral_deg,
                    c.max_dihedral_deg,
                    c.force_constant,
                )
                for c in constraints
            ]
            for constraints in self._torsion_constraints
        ]
        self._native_ff = _batchedForcefield.NativeMMFFBatchedForcefield(
            self._molecules,
            native_properties,
            self._conf_ids,
            distance_constraints,
            position_constraints,
            angle_constraints,
            torsion_constraints,
        )
        self._dirty = False

    def _ensure_built(self) -> None:
        if self._dirty or self._native_ff is None:
            self._build()

    def rebuild(self) -> None:
        """Force a rebuild of the native batched forcefield."""
        self._build()

    def compute_energy(self) -> list[float]:
        """Compute one MMFF energy value per molecule in the batch.

        Note: Per-call Python overhead makes standalone calls slower than
        expected.  Prefer the minimizer for performance-sensitive workloads.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeEnergy()

    def compute_gradients(self) -> list[list[float]]:
        """Compute one flattened 3D gradient vector per molecule in the batch.

        Note: Per-call Python overhead makes standalone calls slower than
        expected.  Prefer the minimizer for performance-sensitive workloads.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeGradients()
