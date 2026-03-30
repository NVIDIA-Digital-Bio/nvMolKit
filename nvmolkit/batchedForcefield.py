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
from typing import TYPE_CHECKING

from nvmolkit import _batchedForcefield  # type: ignore
from nvmolkit._mmff_bridge import default_rdkit_mmff_properties, make_internal_mmff_properties

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties


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
        self._native_ff = None
        self.num_molecules = len(molecules)
        self.data_dim = 3

    def __len__(self) -> int:
        """Return the number of molecules in the batch."""
        return len(self._molecules)

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

    def _normalize_conf_ids(self, conf_id: int | Sequence[int]) -> list[int]:
        if isinstance(conf_id, int):
            return [conf_id for _ in self._molecules]
        if len(conf_id) != len(self._molecules):
            raise ValueError(f"Expected {len(self._molecules)} conf_id values, got {len(conf_id)}")
        return list(conf_id)

    def _copy_mmff_properties(self, mol: "Mol", properties, non_bonded_threshold: float, ignore_interfrag_interactions: bool):
        """Convert public RDKit MMFF properties into nvMolKit's internal transport."""
        source = default_rdkit_mmff_properties(mol) if properties is None else properties
        return make_internal_mmff_properties(
            source,
            non_bonded_threshold=float(non_bonded_threshold),
            ignore_interfrag_interactions=bool(ignore_interfrag_interactions),
        )

    def _build(self) -> None:
        """Build the native batched forcefield from the current state."""
        if not self._molecules:
            self._native_ff = None
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
        self._native_ff = _batchedForcefield.NativeMMFFBatchedForcefield(
            self._molecules,
            native_properties,
            self._conf_ids,
        )

    def _ensure_built(self) -> None:
        if self._native_ff is None:
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
