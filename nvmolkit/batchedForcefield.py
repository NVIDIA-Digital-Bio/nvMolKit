"""Evaluate forcefield energies and gradients for batches of RDKit molecules.

This module provides batched forcefield wrappers. The examples below use
:class:`MMFFBatchedForcefield`, but the module-level description is intended to
remain forcefield-agnostic.

Typical batch usage:

.. code-block:: python

   from rdkit import Chem
   from rdkit.Chem import rdDistGeom
   from nvmolkit.batchedForcefield import MMFFBatchedForcefield

   mol_a = Chem.AddHs(Chem.MolFromSmiles("CCO"))
   mol_b = Chem.AddHs(Chem.MolFromSmiles("CCCC"))

   rdDistGeom.EmbedMultipleConfs(mol_a, numConfs=1)
   rdDistGeom.EmbedMultipleConfs(mol_b, numConfs=1)

   ff = MMFFBatchedForcefield([mol_a, mol_b])
   energies = ff.compute_energy()
   gradients = ff.compute_gradients()

   energy_a = energies[0]
   gradient_b = gradients[1]

If a molecule has multiple conformers, use ``conf_id`` to choose which
conformer is used for evaluation.

The wrapper also supports per-entry settings such as custom property objects,
conformer ids, non-bonded thresholds, and interfragment interaction handling.
These options may be passed either as scalars, which are broadcast to the whole
batch, or as per-molecule sequences:

.. code-block:: python

   ff = MMFFBatchedForcefield(
       [mol_a, mol_b],
       properties=[props_a, props_b],
       conf_id=[0, 3],
       nonBondedThreshold=[100.0, 25.0],
       ignoreInterfragInteractions=[True, False],
   )

Constraint terms are added through :class:`MMFFBatchElement`, returned by
indexing into the batch:

.. code-block:: python

   ff = MMFFBatchedForcefield([mol_a, mol_b])

   ff[0].add_distance_constraint(0, 2, True, 0.2, 0.5, 20.0)
   ff[1].add_torsion_constraint(0, 1, 2, 3, True, 10.0, 20.0, 8.0)

   energies = ff.compute_energy()
   energy_a = energies[0]
   energy_b = energies[1]

.. note::
   Calling :meth:`~MMFFBatchedForcefield.compute_energy` or
   :meth:`~MMFFBatchedForcefield.compute_gradients` individually from Python
   incurs per-call overhead that can dominate the actual GPU work. This wrapper
   is most useful for correctness checks and Python workflows that need batched
   forcefield energies or gradients. For high-throughput optimization, prefer
   the dedicated minimizer APIs.
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
    """Per-entry view for configuring one item in an MMFF batch.

    Retrieve an element with ``ff[i]`` and use it to inspect the number of
    atoms or to add constraints for that batch entry.

    Example:

    .. code-block:: python

       ff = MMFFBatchedForcefield([mol_a, mol_b])

       first = ff[0]
       second = ff[1]

       print(first.num_atoms)
       print(second.num_atoms)

       first.add_position_constraint(0, 0.1, 50.0)
       second.add_distance_constraint(1, 4, False, 2.0, 2.2, 25.0)
       energies = ff.compute_energy()
    """

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
        """Add a distance constraint to this batch entry.

        Args:
            idx1: Index of the first constrained atom.
            idx2: Index of the second constrained atom.
            relative: If ``True``, interpret ``min_len`` and ``max_len`` as
                offsets from the current distance in the selected conformer.
                If ``False``, interpret them as absolute distances in
                Angstroms.
            min_len: Lower bound of the allowed distance range.
            max_len: Upper bound of the allowed distance range.
            force_constant: Constraint force constant.
        """
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
        """Add a position constraint to one atom in this batch entry.

        Args:
            idx: Index of the constrained atom.
            max_displ: Maximum displacement allowed from the current atom
                position before the restraint contributes energy.
            force_constant: Constraint force constant.

        The reference position comes from the conformer selected for this batch
        entry.
        """
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
        """Add an angle constraint to this batch entry.

        Args:
            idx1: Index of the first atom in the angle.
            idx2: Index of the central atom.
            idx3: Index of the third atom in the angle.
            relative: If ``True``, interpret ``min_angle_deg`` and
                ``max_angle_deg`` relative to the current angle in the selected
                conformer. If ``False``, interpret them as absolute angles in
                degrees.
            min_angle_deg: Lower bound of the allowed angle range in degrees.
            max_angle_deg: Upper bound of the allowed angle range in degrees.
            force_constant: Constraint force constant.
        """
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
        """Add a torsion constraint to this batch entry.

        Args:
            idx1: Index of the first atom in the torsion.
            idx2: Index of the second atom in the torsion.
            idx3: Index of the third atom in the torsion.
            idx4: Index of the fourth atom in the torsion.
            relative: If ``True``, interpret ``min_dihedral_deg`` and
                ``max_dihedral_deg`` relative to the current dihedral angle.
                If ``False``, interpret them as absolute dihedral bounds in
                degrees.
            min_dihedral_deg: Lower bound of the allowed dihedral range in
                degrees.
            max_dihedral_deg: Upper bound of the allowed dihedral range in
                degrees.
            force_constant: Constraint force constant.
        """
        self._parent._validate_atom_indices(self._idx, idx1, idx2, idx3, idx4)
        self._parent._torsion_constraints[self._idx].append(
            _TorsionConstraint(
                idx1, idx2, idx3, idx4, relative, min_dihedral_deg, max_dihedral_deg, force_constant
            )
        )
        self._parent._dirty = True


class MMFFBatchedForcefield:
    """Evaluate MMFF energies and gradients for a batch of inputs.

    Each batch entry consists of:

    - one RDKit molecule
    - one selected conformer from that molecule
    - optional per-entry MMFF settings
    - optional constraints added through :class:`MMFFBatchElement`

    Use this class when you want batched MMFF evaluation from Python and need
    one result per entry in the order you provided them.

    Examples:

    .. code-block:: python

       ff = MMFFBatchedForcefield([mol_a, mol_b])
       energies = ff.compute_energy()
       gradients = ff.compute_gradients()

    .. code-block:: python

       ff = MMFFBatchedForcefield(
           [Chem.Mol(mol), Chem.Mol(mol)],
           conf_id=[0, 1],
       )
       energies = ff.compute_energy()

    .. code-block:: python

       ff = MMFFBatchedForcefield(
           [mol_a, mol_b],
           properties=[props_a, None],
           nonBondedThreshold=[100.0, 20.0],
           ignoreInterfragInteractions=[True, False],
       )
       ff[0].add_distance_constraint(0, 4, False, 1.8, 2.2, 50.0)
       ff[1].add_position_constraint(1, 0.1, 100.0)
       gradients = ff.compute_gradients()

    The values returned by :meth:`compute_gradients` are flattened coordinate
    gradients in ``[x0, y0, z0, x1, y1, z1, ...]`` order, one list per batch
    element.
    """

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
            molecules: RDKit molecules to evaluate. The batch output order
                matches this input order.
            properties: RDKit ``MMFFMolProperties`` object, a per-molecule list
                of those objects, or ``None`` to use default MMFF94 settings.
                A single object is broadcast to all molecules.
            conf_id: Conformer id or per-molecule conformer ids to use. A
                scalar is broadcast to the whole batch. To evaluate multiple
                conformers from one molecule, include that molecule multiple
                times in ``molecules`` and provide one conformer id per batch
                entry.
            nonBondedThreshold: Non-bonded cutoff distance or per-molecule
                values.
            ignoreInterfragInteractions: Whether to omit interfragment non-bonded
                interactions, as a scalar or per-molecule list.
        """
        # TODO: Expand this API to evaluate all conformers from each input
        # molecule directly and return nested per-molecule/per-conformer
        # energies and gradients instead of the current one-entry-per-input
        # shape.
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
        """Return a view for one batch entry.

        Example:

        .. code-block:: python

           ff = MMFFBatchedForcefield([mol_a, mol_b, mol_c])
           ff[1].add_angle_constraint(0, 1, 2, True, 5.0, 10.0, 50.0)
           ff[2].add_position_constraint(3, 0.2, 100.0)
        """
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
        """Convert RDKit MMFF properties into the native representation."""
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
        """Build the native forcefield for the current batch settings."""
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
        """Rebuild the forcefield after changing constraints or settings."""
        self._build()

    def compute_energy(self) -> list[float]:
        """Return one MMFF energy per batch entry.

        Returns:
            A list of MMFF energies in batch order.

        Example:

        .. code-block:: python

           energies = ff.compute_energy()
           first_energy = energies[0]
           second_energy = energies[1]

        Note:
            Per-call Python overhead makes standalone calls slower than
            expected. Prefer the minimizer for performance-sensitive workloads.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeEnergy()

    def compute_gradients(self) -> list[list[float]]:
        """Return one flattened 3D gradient vector per batch entry.

        Returns:
            A list with one flattened gradient vector per batch entry. Each
            gradient is ordered as ``[x0, y0, z0, x1, y1, z1, ...]``.

        Example:

        .. code-block:: python

           gradients = ff.compute_gradients()
           grad_atom0_entry0 = gradients[0][0:3]
           grad_atom0_entry1 = gradients[1][0:3]

        Note:
            Per-call Python overhead makes standalone calls slower than
            expected. Prefer the minimizer for performance-sensitive workloads.
        """
        if not self._molecules:
            return []
        self._ensure_built()
        return self._native_ff.computeGradients()
