"""Bridge RDKit MMFF property objects into nvMolKit's internal MMFF config.

This shim exists because RDKit ``Mol`` objects pass cleanly across the Python/C++
extension boundary as ``ROMol*``, but RDKit's Python ``MMFFMolProperties`` object
does not. It is a separate Boost.Python wrapper type owned by RDKit's forcefield
module, so nvMolKit accepts that object at the public Python API and converts it
into its own plain internal ``MMFFProperties`` transport object before calling
into the native extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import weakref

from rdkit.Chem import rdForceFieldHelpers
from rdkit.ForceField import rdForceField as _rdForceField  # noqa: F401

from nvmolkit import _batchedForcefield  # type: ignore

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.ForceField.rdForceField import MMFFMolProperties as RDKitMMFFMolProperties


_DEFAULT_MMFF_SETTINGS = {
    "variant": "MMFF94",
    "dielectric_constant": 1.0,
    "dielectric_model": 1,
    "bond_term": True,
    "angle_term": True,
    "stretch_bend_term": True,
    "oop_term": True,
    "torsion_term": True,
    "vdw_term": True,
    "ele_term": True,
}
_CAPTURED_MMFF_SETTINGS_BY_OBJECT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _normalize_mmff_settings(settings: dict | None) -> dict:
    normalized = dict(_DEFAULT_MMFF_SETTINGS)
    if settings is not None:
        normalized.update(settings)
    return normalized


def capture_mmff_settings(properties: "RDKitMMFFMolProperties", settings: dict | None):
    """Associate explicit MMFF settings with an RDKit MMFF properties object.

    We use this when nvmolkit itself creates/configures a Python
    ``MMFFMolProperties`` object so later conversion back into nvMolKit's
    internal MMFF transport does not depend on RDKit exposing Python getters for
    every setting in a given build.
    """

    _CAPTURED_MMFF_SETTINGS_BY_OBJECT[properties] = _normalize_mmff_settings(settings)
    return properties


def default_rdkit_mmff_properties(mol: "Mol"):
    """Create default RDKit MMFF properties and capture their default settings."""

    properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    if properties is None:
        raise ValueError("RDKit could not create MMFF properties for molecule")
    return capture_mmff_settings(properties, None)


def extract_mmff_settings(properties: "RDKitMMFFMolProperties") -> dict:
    """Return MMFF settings for an RDKit MMFF properties object.

    Captured settings take precedence. If we did not create/configure the object
    ourselves, fall back to RDKit getter methods when the current build exposes
    them.
    """

    captured = _CAPTURED_MMFF_SETTINGS_BY_OBJECT.get(properties)
    if captured is not None:
        return dict(captured)

    getter_candidates = {
        "variant": ("GetMMFFVariant", "getMMFFVariant"),
        "dielectric_constant": ("GetMMFFDielectricConstant", "getMMFFDielectricConstant"),
        "dielectric_model": ("GetMMFFDielectricModel", "getMMFFDielectricModel"),
        "bond_term": ("GetMMFFBondTerm", "getMMFFBondTerm"),
        "angle_term": ("GetMMFFAngleTerm", "getMMFFAngleTerm"),
        "stretch_bend_term": ("GetMMFFStretchBendTerm", "getMMFFStretchBendTerm"),
        "oop_term": ("GetMMFFOopTerm", "getMMFFOopTerm"),
        "torsion_term": ("GetMMFFTorsionTerm", "getMMFFTorsionTerm"),
        "vdw_term": ("GetMMFFVdWTerm", "getMMFFVdWTerm"),
        "ele_term": ("GetMMFFEleTerm", "getMMFFEleTerm"),
    }
    extracted = {}
    for key, names in getter_candidates.items():
        getter = None
        for name in names:
            if hasattr(properties, name):
                getter = getattr(properties, name)
                break
        if getter is None:
            raise TypeError(
                "Could not read MMFF settings from the supplied RDKit MMFFMolProperties object. "
                "Use an object created via rdForceFieldHelpers.MMFFGetMoleculeProperties() and "
                "configured before passing it to nvmolkit."
            )
        extracted[key] = getter()
    return _normalize_mmff_settings(extracted)


def make_internal_mmff_properties(
    properties: "RDKitMMFFMolProperties",
    *,
    non_bonded_threshold: float,
    ignore_interfrag_interactions: bool,
):
    """Convert an RDKit MMFF properties object into nvMolKit's internal transport.

    Unlike RDKit ``Mol`` objects, RDKit's Python ``MMFFMolProperties`` wrapper is
    not passed directly into nvMolKit's extension module. The native code instead
    receives this plain internal ``MMFFProperties`` object with the RDKit settings
    copied onto it.
    """

    settings = extract_mmff_settings(properties)
    internal = _batchedForcefield.MMFFProperties()
    internal.variant = str(settings["variant"])
    internal.dielectricConstant = float(settings["dielectric_constant"])
    internal.dielectricModel = int(settings["dielectric_model"])
    internal.nonBondedThreshold = float(non_bonded_threshold)
    internal.ignoreInterfragInteractions = bool(ignore_interfrag_interactions)
    internal.bondTerm = bool(settings["bond_term"])
    internal.angleTerm = bool(settings["angle_term"])
    internal.stretchBendTerm = bool(settings["stretch_bend_term"])
    internal.oopTerm = bool(settings["oop_term"])
    internal.torsionTerm = bool(settings["torsion_term"])
    internal.vdwTerm = bool(settings["vdw_term"])
    internal.eleTerm = bool(settings["ele_term"])
    return internal
