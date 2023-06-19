import numpy as np
import time

import numpy as np
import time
import torch


def dmp(Pts, n, m, grid):
    dense_map_start = time.time()
    ng = 2 * grid + 1

    mX = np.zeros((m, n)) + float("inf")
    mY = np.zeros((m, n)) + float("inf")
    mD = np.zeros((m, n))
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]

    print(KmX.shape)

    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s

    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    dense_map_end = time.time()
    total_time = dense_map_end - dense_map_start

    print(f"Total time old: {total_time}")
    return out


def dmpt(Pts, n, m, grid):
    dense_map_start = time.time()

    ng = 2 * grid + 1

    device = 'cuda'
    mX = torch.full((m, n), float("inf"), device=device)
    mY = torch.full((m, n), float("inf"), device=device)
    mD = torch.zeros((m, n), device=device)

    idx1 = Pts[1].to(torch.long)
    idx2 = Pts[0].to(torch.long)
    mX[idx1, idx2] = Pts[0].to(torch.float) - torch.round(Pts[0]).to(torch.float)
    mY[idx1, idx2] = Pts[1].to(torch.float) - torch.round(Pts[1]).to(torch.float)
    mD[idx1, idx2] = Pts[2].to(torch.float)

    i_offsets = torch.arange(ng, device=device).view(-1, 1, 1, 1)
    j_offsets = torch.arange(ng, device=device).view(1, -1, 1, 1)

    i_indices = i_offsets + torch.arange(m - ng, device=device).view(1, 1, -1, 1)
    j_indices = j_offsets + torch.arange(n - ng, device=device).view(1, 1, 1, -1)

    KmX = mX[i_indices, j_indices] - (grid + i_offsets - 1)
    KmY = mY[i_indices, j_indices] - (grid + i_offsets - 1)
    KmD = mD[i_indices, j_indices]

    s = 1 / torch.sqrt(KmX * KmX + KmY * KmY)
    Y = torch.sum(s * KmD, dim=(0, 1))
    S = torch.sum(s, dim=(0, 1))

    S[S == 0] = 1

    out = torch.zeros((m, n), device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S

    dense_map_end = time.time()
    total_time = dense_map_end - dense_map_start

    print(f"Total time with PyTorch: {total_time}")
    return out

# You can then call dd_optimized with the same arguments as dd
# Example:
# dd_optimized(Pts, n, m, grid)

Pts = np.load("../Pts.npy")

tPts = torch.tensor(Pts).cuda()
out = dmpt(tPts, 1600, 900, 13)

print("Sum new", torch.sum(out))

out2 = dmp(Pts, 1600, 900, 13)
print("Sum old", np.sum(out2))

