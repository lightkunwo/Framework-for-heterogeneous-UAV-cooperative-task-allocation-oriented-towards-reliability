from math import comb


class ConsecutiveKOutOfN:


    def __init__(self, n, k):

        self.n = n
        self.k = k
        self.cache = {}

    def compute_N(self, i, k, n):

        if i == 0:
            return 1
        if i >= n:
            return 0
        if k == 1:
            return 0
        if i * k > n + i - 1:
            return 0


        cache_key = (i, k, n)
        if cache_key in self.cache:
            return self.cache[cache_key]


        if k == 2:
            if i > (n + 1) // 2:
                result = 0
            else:
                result = comb(n - i + 1, i)


        elif k == 3:
            result = 0
            h_min = max(i - (n - i + 1), 0)
            h_max = min(i // 2, n - i + 1)
            for h in range(h_min, h_max + 1):
                if n - i + 1 >= h and n - i + 1 - h >= i - 2 * h >= 0:
                    result += comb(n - i + 1, h) * comb(n - i + 1 - h, i - 2 * h)

        # k>3 general case
        else:
            result = self.compute_M(i, k - 1, n - i + 1)

        # Cache result
        self.cache[cache_key] = result
        return result

    def compute_M(self, i, k_minus_1, m):

        if i == 0:
            return 1
        if i < 0 or m < 0:
            return 0
        if k_minus_1 == 1:
            return 0

        # k-1=2 special case
        if k_minus_1 == 2:
            if i > (m + 1) // 2:
                return 0
            else:
                return comb(m - i + 1, i) if m - i + 1 >= i >= 0 else 0

        # Recursive computation
        result = 0
        for h in range(min(m, i // k_minus_1) + 1):
            if m >= h and i >= k_minus_1 * h:
                sub_result = self.compute_M(i - k_minus_1 * h, k_minus_1 - 1, m - h)
                result += comb(m, h) * sub_result
        return result

    def calculate_swarm_reliability(self, R_U):

        if isinstance(R_U, (list, tuple)):
            import numpy as np
            R_U = np.mean(R_U)

        R_swarm = 0.0

        for i in range(self.n + 1):
            N_i = self.compute_N(i, self.k, self.n)

            if N_i > 0:

                prob = (R_U ** (self.n - i)) * ((1 - R_U) ** i)
                R_swarm += N_i * prob

        return R_swarm

    def print_N_table(self):

        print(f"\n{'=' * 60}")
        print(f"N(i, k={self.k}, n={self.n}) Combination Table")
        print(f"{'=' * 60}")
        print(f"{'i':<5} {'N(i,k,n)':<15} {'Description'}")
        print(f"{'-' * 60}")

        for i in range(self.n + 1):
            N_i = self.compute_N(i, self.k, self.n)
            desc = "Valid" if N_i > 0 else "Has consecutive failures"
            print(f"{i:<5} {N_i:<15} {desc}")
        print(f"{'=' * 60}\n")


# Test function
if __name__ == "__main__":
    from math import exp

    print("Testing Consecutive-k-out-of-n System")
    print("=" * 60)

    n = 6
    k = 2

    system = ConsecutiveKOutOfN(n, k)

    # Print N(i,k,n) table
    system.print_N_table()

    # Calculate reliability
    alpha = 0.0001
    t = 100
    R_U = exp(-alpha * t)

    R_swarm = system.calculate_swarm_reliability(R_U)

    print(f"Individual UAV reliability: {R_U:.6f}")
    print(f"Swarm reliability: {R_swarm:.6f}")
    print(f"Improvement: {R_swarm / R_U:.2f}x")