import numpy as np

np.random.seed(200)

'''
Params:
point: Training Data
M: Total of both 
D: Number of features
c: Inversely proportional to the amount of mistakes we will allow
tol: Tolerance for the KKT conditions
prog_margin: Margin for determining whether the two langrange multipliers made any positive progress
clip_padding: Padding for clipping the new/optimized multiplier to the bounds 
'''

class SMO_GAUSSIAN():
    def __init__(self, point: np.float64, target: np.float64, M: int, D: int, 
                 c: np.float64=1, tol: np.float64=1e-5, 
                 prog_margin: np.float64=1e-5, clip_padding:np.float64=1e-5, log: bool=False):
        # Configs
        self.M:              int         = M
        self.D:              int         = D
        self.point:          np.int32    = point
        self.target :        np.float64  = target
        self.c:              np.float64  = c
        self.tol:            np.float64  = tol
        self.prog_margin:    np.float64  = prog_margin
        self.clip_padding:   np.float64  = clip_padding
        self.sigma:          int         = 1
        self.max_iter:       int         = 10000000
        self.log:            bool        = log

        # Important Values
        self.alphs:     np.float64 = np.zeros(shape=(self.M), dtype=np.float64)
        self.err_cache: np.float64 = np.zeros(shape=(self.M), dtype=np.float64)
        self.B:         np.float64 = 0

        #self.__initialize_dot_cache()

    def smo_train(self) -> dict:
        self.__examine_all: bool = True
        num_changed: int = 0
        total_iter: int = 0

        while num_changed > 0 or self.__examine_all:
            # print("choosing first multiplier")
            if total_iter >= self.max_iter:
                print("Exceeded max iterations")
                return
        
            if self.log and total_iter%2 == 0:
                print("total iterations: ", total_iter)

            num_changed = 0

            # sorted non_bound_alphs using insertion sort
            non_bound_alphs: list = []
            for i in range(self.M):
                # non-bound
                if (self.alphs[i] != 0 and self.alphs[i] != self.c):
                    # insert
                    inserted: bool = False
                    for j in range(len(non_bound_alphs)):
                        if (self.alphs[non_bound_alphs[j]] >= self.alphs[i]):
                            non_bound_alphs.insert(j, i)
                            inserted = True
                            break
                    
                    if not inserted:
                        non_bound_alphs.append(i)
            
            non_bound_alphs = np.array(non_bound_alphs, dtype=np.int32)

            if self.__examine_all:
                for i in range(self.M):
                    num_changed += self.__examine_a(i, False, non_bound_alphs)
            else:
                for i in non_bound_alphs:
                    num_changed += self.__examine_a(i, True, non_bound_alphs)

            if self.__examine_all:
                self.__examine_all = False
            elif num_changed == 0:
                self.__examine_all = True

            total_iter += 1
    

    def __examine_a(self, i2: int, non_bound: bool, non_bound_alphs: np.int64) -> bool:
        non_b_len: int = len(non_bound_alphs)
        E2: np.float64 = self.err_cache[i2] if non_bound else self.__compute_svm_err(i2)
        r2: np.float64 = E2 * self.target[i2]
        alph2: np.float64 = self.alphs[i2]

        if ((r2 < -self.tol and alph2 < self.c) or (r2 > self.tol and alph2 > 0)):
            # print("choosing second multipier")
            if (non_b_len > 1):
                # Second heuristic using optimal err for step estimation
                positive: bool = (self.err_cache[i2] > 0)

                i1: int = i2
                if (positive):
                    for idx in non_bound_alphs:
                        if (idx == i2):
                            continue

                        i1 = idx
                else:
                    for idx in range(non_b_len - 1, -1, -1):
                        if (non_bound_alphs[idx] == i2):
                            continue

                        i1 = non_bound_alphs[idx]
    
            # take non-bound alps
            # because we are sorting non_b_alphs, we can consider this as some kind of randomization
            if non_b_len > 0:
                start: int = np.random.randint(size=(1), low=0, high=non_b_len)[0]
                for i in range(non_b_len):
                    pos: int = (start + i)%non_b_len
                    if (self.__take_step(non_bound_alphs[pos], i2, non_bound_alphs)):
                        return 1
            
            # loop through entire training set
            start: int = np.random.randint(size=(1), low=0, high=self.M)[0]
            for i in range(self.M):
                if (self.__take_step((start + i) % self.M, i2, non_bound_alphs)):
                    return 1
        
        # print("already satisfy kkt conditions")
        return 0
        

    def __take_step(self, i1: int, i2: int, non_bound_alphs: np.int64, log: bool=False) -> bool:
        #print(f"Taking step for {i1} and {i2}")
        if (i1 == i2):
            #print("Positions equal")
            return 0
        
        non_bound: bool = self.alphs[i1] != 0 and self.alphs[i1] != self.c

        E1:    np.float64  = self.err_cache[i1] if non_bound else self.__compute_svm_err(i1)
        E2:    np.float64  = self.err_cache[i2] 
        y1:    int         = self.target[i1]
        y2:    int         = self.target[i2]
        alph1: np.float64  = self.alphs[i1]
        alph2: np.float64  = self.alphs[i2]
        s:     int         = y1 * y2

        # Computation for L and H
        if (y1 == y2):
            L: np.float64 = max(0, alph1 + alph2 - self.c)
            H: np.float64 = min(self.c, alph1 + alph2)
        else:
            L: np.float64 = max(0, alph2 - alph1)
            H: np.float64 = min(self.c, self.c + alph2 - alph1)

        if (L == H):
            #print("L and H equal")
            return 0
        
        K11: np.float64 = self.__kernel_gaussian(self.point[i1], self.point[i1])
        K22: np.float64 = self.__kernel_gaussian(self.point[i2], self.point[i2])
        K12: np.float64 = self.__kernel_gaussian(self.point[i1], self.point[i2])
        eta: np.float64 = K11 + K22 - 2*K12

        if (eta > 0):
            alph2_new: np.float64 = (alph2) + (y2*(E1 - E2)/eta)
            if (alph2_new <= L):
                alph2_new = L
            elif (alph2_new >= H):
                alph2_new = H
        else:
            v1: np.float64 = E1 - alph1*y1*K11 - alph2*y2*K12
            v2: np.float64 = E2 - alph1*y1*K12 - alph2*y2*K22
            zeta: np.float64 = alph1*y1 + alph2*y2
            Lobj: np.float64 = L*(1-s) + zeta*s*L*K11 - (.05*(L**2)*(K11 + K22)) - (zeta - s*L)*s*L*K12 + (v1 - v2)*y2*L
            Hobj: np.float64 = H*(1-s) + zeta*s*H*K11 - (.05*(H**2)*(K11 + K22)) - (zeta - s*H)*s*H*K12 + (v1 - v2)*y2*H

            if (Lobj < Hobj - self.prog_margin):
                alph2_new: np.float64 = L
            elif (Lobj > Hobj + self.prog_margin):
                alph2_new: np.float64 = H    
            else:
                alph2_new: np.float64 = alph2


        if (abs(alph2_new - alph2) < (self.prog_margin * (alph2_new + alph2 + self.prog_margin))):
            #print("Bad progress")
            return 0
        
        alph1_new: np.float64 = alph1 + (s*(alph2 - alph2_new))

        # clip
        if (alph1_new < self.clip_padding):
            alph1_new = 0
        elif (alph1_new > self.c - self.clip_padding):
            alph1_new = self.c

        # copy of B before changing it
        b: int = self.B 
        # update tresholds
        b1: np.float64 = self.B - E1 - y1*K11*(alph1_new - alph1) - y2*K12*(alph2_new - alph2)
        b2: np.float64 = self.B - E2 - y1*K12*(alph1_new - alph1) - y2*K22*(alph2_new - alph2)
        
        if (0 < alph1_new < self.c):
            self.B = b1
        elif (0 < alph2_new < self.c):
            self.B = b2
        else:
            self.B = (b1 + b2)/2

        # update alphs
        self.alphs[i1], self.alphs[i2] = alph1_new, alph2_new

        # update err_cache
        self.err_cache[i1], self.err_cache[i2] = 0, 0
        for i in non_bound_alphs:
            if (i == i1 or i == i2):
                continue
            
            K1k: np.float64 = self.__kernel_gaussian(self.point[i1], self.point[i])
            K2k: np.float64 = self.__kernel_gaussian(self.point[i2], self.point[i])

            self.err_cache[i] += y1*K1k*(alph1_new - alph1) + y2*K2k*(alph2_new - alph2) + (self.B - b)
        
        return 1

    def __compute_svm_err(self, x: int) -> np.float64:
        fx: np.float64 = self.__obj_x(self.point[x])
        self.err_cache[x] = fx - self.target[x]
        return self.err_cache[x]     


    def __obj_x(self,val: np.float64) -> np.float64:
        fx: np.float64 = (self.alphs * self.target) @ self.__kernel_gaussian(self.point, val) + self.B
        return fx


    def __kernel_gaussian(self, x1, x2):
        if np.ndim(x1) == 1 and np.ndim(x2) == 1:
            return np.exp(-(np.linalg.norm(x1-x2,2))**2/(2*self.sigma**2))
        elif(np.ndim(x1)>1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2)>1):
            return np.exp(-(np.linalg.norm(x1-x2, 2, axis=1)**2)/(2*self.sigma**2))
        elif np.ndim(x1) > 1 and np.ndim(x2) > 1 :
            return np.exp(-(np.linalg.norm(x1[:, np.newaxis] \
                             - x2[np.newaxis, :], 2, axis = 2) ** 2)/(2*self.sigma**2))
        
        return 0.
    

    def accuracy(self) -> np.float64:
        correct: float = 0.0

        for i in range(self.M):
            fx: np.float64 = self.__obj_x(self.point[i])
            correct += (self.target[i] * fx > 0)
        
        return correct / self.M


    def predict(self, x: np.float64, new: bool = False) -> np.float64:
        res: np.float64 = np.zeros(shape=(len(x)), dtype=np.float64)

        for i in range(len(x)):
            fx: np.float64 = self.__obj_x(x[i])
            res[i] = fx
        
        return res