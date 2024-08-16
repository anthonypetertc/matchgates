import numpy as np
import scipy.linalg as la
import MatchGates.operators as ops
from MatchGates.operators import X, Y, I, Z
from functools import reduce
from numbers import Real

class MatchGate():
    def __init__(self, param_dict):
        assert param_dict.keys() == set(["z1", "z2", "xx", "yy", "xy", "yx"])
        self.param_dict = param_dict

        H = param_dict["z1"] * ops.Z1 + param_dict["z2"] * ops.Z2 + param_dict["xx"] * ops.XX \
            + param_dict["yy"] * ops.YY + param_dict["xy"] * ops.XY + param_dict["yx"] * ops.YX
        self.gate = np.round(la.expm(1j * H), 11)

    @classmethod
    def fromUnitary(self, U):
        U = np.round(U, 11)
        assert U.shape == (4, 4)
        assert np.allclose(U[0, 1:3], [0, 0])
        assert np.allclose(U[1:3, 0], [0, 0])
        assert np.allclose(U[3, 1:3], [0, 0])
        assert np.allclose(U[1:3, 3], [0, 0])
        A = np.zeros((2,2), dtype=complex)
        A[0, 0] = U[0, 0]
        A[0, 1] = U[0, 3]
        A[1, 0] = U[3, 0]
        A[1, 1] = U[3, 3]
        B = U[1:3, 1:3]
        assert np.isclose(la.det(A), la.det(B)) 

        H = -1j * la.logm(U)
        assert np.allclose(H.conj().transpose(), H)
        param_dict = {}
        param_dict["z1"] = 0.25 * np.trace(ops.Z1 @ H)
        param_dict["z2"] = 0.25 * np.trace(ops.Z2 @ H)
        param_dict["xx"] = 0.25 * np.trace(ops.XX @ H)
        param_dict["yy"] = 0.25 * np.trace(ops.YY @ H)
        param_dict["xy"] = 0.25 * np.trace(ops.XY @ H)
        param_dict["yx"] = 0.25 * np.trace(ops.YX @ H)


        return MatchGate(param_dict)
    
    @classmethod
    def fromAB(self, A, B):
        assert A.shape == (2, 2)
        assert B.shape == (2, 2)
        assert np.isclose(la.det(A), la.det(B))
        U = np.zeros((4,4), dtype=complex)
        U[0, 0] = A[0, 0]
        U[0, 3] = A[0, 1]
        U[3, 0] = A[1, 0]
        U[3, 3] = A[1, 1]
        U[1,1] = B[0, 0]
        U[1, 2] = B[0, 1]
        U[2, 1] = B[1, 0]
        U[2, 2] = B[1, 1]
        return MatchGate.fromUnitary(U)

    def swap(self):
        param_dict = self.param_dict
        new_param_dict = {}
        new_param_dict["z1"] = param_dict["z2"]
        new_param_dict["z2"] = param_dict["z1"]
        new_param_dict["xx"] = param_dict["xx"]
        new_param_dict["yy"] = param_dict["yy"]
        new_param_dict["xy"] = param_dict["yx"]
        new_param_dict["yx"] = param_dict["xy"]
        return MatchGate(new_param_dict)
    
    @classmethod
    def randomMatchGate(self, seed=None):
        param_dict = {}
        if seed is not None:
            np.random.seed(seed)
        random_numbers = np.random.rand(6)
        param_dict["z1"] = random_numbers[0]
        param_dict["z2"] = random_numbers[1]
        param_dict["xx"] = random_numbers[2]
        param_dict["yy"] = random_numbers[3]
        param_dict["xy"] = random_numbers[4]
        param_dict["yx"] = random_numbers[5]
        return MatchGate(param_dict)
    @classmethod
    def empty_param_dict(self):
        return {"z1":0, "z2":0, "xx":0, "yy":0, "xy":0, "yx":0}
    
    def is_single_qubit(self):
        xx = self.param_dict["xx"]
        yy = self.param_dict["yy"]
        xy = self.param_dict["xy"]
        yx = self.param_dict["yx"]
        if np.allclose([xx, yy, xy, yx], 0, atol=1e-11):
            return True
        else:
            return False




class AppliedMatchGate():
    def __init__(self, mg: MatchGate, n_qubits: int, acts_on: tuple):
        assert len(acts_on) == 2
        if acts_on[1] < acts_on[0]:
            mg = mg.swap()
            acts_on = sorted(acts_on)
        elif acts_on[1] == acts_on[0]:
            raise ValueError("Gate must act on two distinct qubits.")
        assert acts_on[1] == acts_on[0] + 1, "Matchgates must be nearest neighbour."
        assert acts_on[1] < n_qubits
        self.mg = mg
        self.n_qubits = n_qubits
        self.acts_on = acts_on
        k = acts_on[0]
        alpha = np.zeros((4, 4), dtype=complex)
        alpha[1, 2] =  mg.param_dict["xx"] /2
        alpha[0, 1] =  mg.param_dict["z1"] /2
        alpha[2, 3] =  mg.param_dict["z2"] /2
        alpha[0, 3] = -mg.param_dict["yy"]/2
        alpha[0, 2] = -mg.param_dict["yx"]/2
        alpha[1, 3] = mg.param_dict["xy"]/2

        alpha[2, 1] = -alpha[1, 2]
        alpha[1, 0] = -alpha[0, 1]
        alpha[3, 2] = -alpha[2, 3]
        alpha[3, 0] = -alpha[0, 3]
        alpha[2, 0] = -alpha[0, 2]
        alpha[3, 1] = -alpha[1, 3]
        h = np.zeros((2*n_qubits, 2*n_qubits), dtype=complex)
        h[2*k:2*k+4, 2*k:2*k+4] = alpha
        T = la.expm(4 * h)
        self.T = T


    def convert_to_unitary(self):
        n = self.n_qubits
        (a, b) = self.acts_on
        U = self.mg.gate
        startlist = [ops.I] * a
        endlist = [ops.I] * (n-b-1)
        return reduce(np.kron, startlist + [U] + endlist) 



class SingleGateNoise():
    def __init__(self, noise_list: list[tuple[MatchGate, Real]], id_prob: Real):
        probabilities = [noise[1] for noise in noise_list]
        kops = [noise[0] for noise in noise_list]
        assert np.isclose(sum(probabilities) + id_prob, 1), "Probabilities must sum to 1."
        for op in kops:
            assert isinstance(op, MatchGate), "Kraus operators must be scaled Matchgates."
        self.probabilities = probabilities
        self.ops = kops
        self.id_prob = id_prob
    
    @classmethod
    def matchgate_depolarizing(self, p):
        assert p < 1
        kraus_ops = [np.kron(X, X), np.kron(Y, Y), np.kron(Z, I),  np.kron(I, Z), np.kron(Z, Z), np.kron(X, Y), np.kron(Y, X)]
        mg_list = []
        for kraus in kraus_ops:
            mg_list.append(MatchGate.fromUnitary(kraus))
        return SingleGateNoise([(mg, p/7) for mg in mg_list], 1-p)
    
    @classmethod
    def trivial_noise(self):
        kraus = np.kron(X, X)
        mg = MatchGate.fromUnitary(kraus)
        return SingleGateNoise([(mg, 0)], 1)


class NoisyAppliedMatchGate():
    def __init__(self, amg: AppliedMatchGate, noise: SingleGateNoise):
        self.amg = amg
        self.noise= noise



