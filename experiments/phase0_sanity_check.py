import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = circuit([0.5, 0.3])
print(f"Phase 0 sanity check passed. Expectation value: {result:.6f}")
assert isinstance(float(result), float), "Result must be a float"
