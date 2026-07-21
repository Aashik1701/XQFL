import pennylane as qml
import numpy as np


def angle_encoding(x, wires):
    """Encode each feature as an RY rotation angle on its qubit.

    Features should be pre-scaled to [0, pi] so rotations span the full Bloch sphere.
    """
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)


def amplitude_encoding(x, wires):
    """Pack the feature vector into the amplitudes of the quantum state.

    The n_qubits input features are zero-padded to 2^n_qubits, then L2-normalised
    to satisfy the unit-norm requirement of a valid quantum state.

    Gradient flows through subsequent VQC weights (backprop) but NOT through x
    (data is always detached before StatePrep — this is intentional).

    Note: on a simulator StatePrep has circuit depth 1; on real hardware the
    depth is O(2^n_qubits).  Circuit-depth comparisons are simulator-level only.
    """
    n_qubits = len(wires)
    dim = 2 ** n_qubits
    # Convert from torch tensor or ndarray to a plain float numpy array
    if hasattr(x, 'detach'):
        x_np = x.detach().cpu().numpy().astype(float)
    else:
        x_np = np.asarray(x, dtype=float)
    vec = np.zeros(dim, dtype=complex)
    vec[:min(len(x_np), dim)] = x_np[:dim]
    norm = np.linalg.norm(vec)
    vec = vec / norm if norm > 1e-12 else np.eye(1, dim, 0, dtype=complex).flatten()
    qml.StatePrep(vec, wires=wires)


def reuploading_encoding(x, wires, n_layers=2):
    """Data re-uploading: re-inject the same feature vector at every layer.

    Each upload applies RY(x_i) to each qubit followed by a ring of CNOT entanglers.
    Increases expressivity by embedding data at multiple depths in the circuit.
    The caller may bind n_layers via functools.partial when it needs to differ from
    the VQC's n_layers.
    """
    for _ in range(n_layers):
        for i, wire in enumerate(wires):
            qml.RY(x[i], wires=wire)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def iqp_encoding(x, wires):
    """Instantaneous Quantum Polynomial (IQP) encoding.

    H -> RZ(x_i) -> IsingZZ(x_i * x_j) for all pairs (i,j) -> H
    Creates feature correlations through entanglement before the variational ansatz.
    """
    n = len(wires)
    for wire in wires:
        qml.Hadamard(wires=wire)
    for i in range(n):
        qml.RZ(x[i], wires=wires[i])
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ(x[i] * x[j], wires=[wires[i], wires[j]])
    for wire in wires:
        qml.Hadamard(wires=wire)
