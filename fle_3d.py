import numpy as np
import scipy.special as spl
import scipy.sparse as spr
from scipy.fft import dct, idct
import finufft
from scipy.io import loadmat
import os


class FLEBasis3D:
    #
    #   N               basis for N x N x N volumes
    #   bandlimit       bandlimit parameter (scaled so that N is max suggested)
    #   eps             requested relative precision in evaluate and evaluate_t
    #   expand_eps      requested approximate relative precision in the expand method (if not specified, pre-tuned values are used)
    #   expand_alpha    requested step-size in the expand method (if not specified, pre-tuned values are used)
    #   expand_rel_tol  requested relative tolerance in the expand method (if not specified, pre-tuned values are used)
    #   maxitr          maximum number of iterations for the expand method (if not specified, pre-tuned values are used)
    #   maxfun          maximum number of basis functions to use (if not specified, which is the default, the number implied by the choice of bandlimit is used)
    #   max_l           use only indices l <= max_l, if not None (default).
    #   mode            choose either "real" or "complex" (default) output
    #   force_real      If true, get a speedup by a factor 2 by enforcing that the source (for evaluate_t) or target (for evaluate)
    #                   is real. To reproduce the tables and figures of the paper, set this to True.
    #   sph_harm_solver solver to use for spherical harmonics expansions.
    #                   Choose either "nvidia_torch" (default) or "FastTransforms.jl".
    #   reduce_memory   If True, reduces the number of radial points in defining
    #                   NUFFT grids, and does an alternative interpolation to
    #                   compensate. To reproduce the tables and figures of the
    #                   paper, set this to False. 
    def __init__(
        self,
        N,
        bandlimit,
        eps,
        expand_eps=1e-4,
        expand_alpha=0.5,
        expand_rel_tol=1e-2,
        maxitr=None,
        maxfun=None,
        max_l=None,
        mode="complex",
        force_real=False,
        sph_harm_solver="nvidia_torch",
        reduce_memory=True
    ):
        realmode = mode == "real"
        complexmode = mode == "complex"
        assert realmode or complexmode

        self.complexmode = complexmode
        self.force_real = force_real
        self.sph_harm_solver = sph_harm_solver
        self.reduce_memory = reduce_memory
        self.eps = eps

        self.expand_alpha = expand_alpha
        self.expand_rel_tol = expand_rel_tol
        self.expand_eps = expand_eps

        #sets numsparse and maxitr heuristically
        numsparse = 32
        if not maxitr:
            tmp = 1 + int(6 * np.log2(N))

        if eps >= 1e-10:
            numsparse = 32 
            if not maxitr:
                tmp = 1 + int(4 * np.log2(N))

        if eps >= 1e-7:
            numsparse = 16
            if not maxitr:
                tmp = 1 + int(3 * np.log2(N))

        if eps >= 1e-4:
            numsparse = 8
            if not maxitr:
                tmp = 1 + int(np.log2(N))
 
        if not maxitr:
            maxitr = tmp

        #sets maxitr heuristically
        maxitr = max(maxitr,int(np.log(1/expand_eps)/expand_alpha)+1)

        self.maxitr = maxitr

        self.W = self.precomp(
            N,
            bandlimit,
            eps,
            maxitr,
            numsparse,
            maxfun=maxfun,
            max_l=max_l
        )

    def precomp(
        self,
        N,
        bandlimit,
        eps,
        maxitr,
        numsparse,
        maxfun=None,
        max_l=None
    ):

        # Original dimension
        self.N1 = N
        # If dimensions are odd, add one (will be zero-padding)
        N = N + (N % 2)

        # Either use maxfun or estimate an upper bound on maxfun based on N
        if maxfun:
            ne = maxfun
        else:
            # approximate number of pixels in the ball of radius N/2
            ne = int(N**3 * np.pi / 6)

        ls, ks, ms, mds, lmds, cs, ne = self.lap_eig_ball(
            ne, bandlimit, max_l=max_l
        )

        self.lmds = lmds
        self.ne = ne
        self.cs = cs
        self.ks = ks
        self.ls = ls
        self.ms = ms

        self.c2r = self.precomp_transform_complex_to_real(ms)
        self.r2c = spr.csr_matrix(self.c2r.transpose().conj())

        # max k,l,m
        kmax = np.max(np.abs(ks))
        lmax = np.max(np.abs(ls))
        mmax = np.max(np.abs(ms))

        self.kmax = kmax
        self.lmax = lmax
        self.mmax = mmax

        epsdis = (
            eps
            / 4
            / (
                np.pi**2
                * (3 / 2) ** (1 / 4)
                * (3 + np.pi / 2 * np.log(5.3 * N))
            )
        )

        Q = int(np.ceil(max(5.3 * N, np.log2(1 / epsdis))))

        tmp = 1 / (np.sqrt(4 * np.pi))

        for Q2 in range(1, Q):
            tmp = tmp / Q2 * ((np.sqrt(3) * np.pi / 16) ** (2 / 3) * (N + 1))
            if tmp < epsdis:
                break

        n_radial = int(Q2)

        if self.reduce_memory:
            n_radial = int(np.ceil(int(Q2) * 0.65))

        S = int(
            max(
                np.ceil(
                    2 * np.exp(1) * 6 ** (1 / 3) * np.pi ** (2 / 3) * (N // 2)
                ),
                4 * np.log2(27.6 / epsdis),
            )
        )

        for S2 in range(1, S):
            tmp = np.exp(1) * lmds[-1] / (2 * (S2 // 2 + 1) + 3)
            tmp2 = (
                28
                / 27
                * np.sqrt(2 * self.lmax + 1)
                * (np.exp(1) * lmds[-1]) ** (3 / 2)
                * (tmp ** (S2 // 2 - 1 / 2))
                * 1
                / (1 - tmp)
            )

            if tmp2 < epsdis:
                if tmp < 1:  
                    break

        S = max(S2, 2 * self.lmax, 18)

        epsnufH = max(
            eps
            / (2 * np.pi ** (3 / 2) * (3 / 2) ** (1 / 4))
            / (2 + np.pi / 2 * np.log(Q)),
            1.1e-15,
        )
        epsnuf = max(
            1
            / 4
            * np.sqrt(np.pi)
            * eps
            / (np.pi ** (2) * (3 / 2) ** (1 / 4))
            / (3 + np.pi / 2 * np.log(Q)),
            1.1e-15,
        )


        if self.sph_harm_solver == "nvidia_torch":
            import torch
            import torch_harmonics as th

            self.step2 = self.step2_torch
            self.step2_H = self.step2_H_torch
            self.torch = torch
            self.th = th


            n_phi = S + 1
             
            n_theta = S  

            ##### Added this to make the symmetry trick in step 1 work
            if n_phi % 2 == 1:
                n_phi += 1

            if self.force_real:
                phi = 2 * np.pi * np.arange(n_phi // 2) / n_phi
            else:
                phi = 2 * np.pi * np.arange(n_phi) / n_phi

            # grid = "legendre-gauss"
            grid = "equiangular"
            if grid == "equiangular":
                cost, weights = self.th.quadrature.clenshaw_curtiss_weights(
                    n_theta, -1, 1
                )
                theta = np.flip(np.arccos(cost))
            elif grid == "legendre-gauss":
                cost, weights = self.th.quadrature.legendre_gauss_weights(
                    n_theta, -1, 1
                )
                theta = np.flip(np.arccos(cost))

            device = "cpu"
            # device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

            sht = self.th.RealSHT(
                n_theta, n_phi, lmax=self.lmax + 1, grid=grid, csphase=True
            ).to(device)
            isht = self.th.InverseRealSHT(
                n_theta,
                n_phi,
                lmax=self.lmax + 1,
                mmax=self.lmax + 1,
                grid=grid,
                csphase=True,
            )

            self.sht = sht
            self.isht = isht
            self.device = device
            self.weights = weights

        elif self.sph_harm_solver == "FastTransforms.jl":
            from juliacall import Main as jl

            jl.seval("using FastSphericalHarmonics")
            jl.seval("using FastTransforms")
            jl.seval("using LinearAlgebra")
            jl.seval("using LinearAlgebra: ldiv! as ldiv")

            F = jl.FastSphericalHarmonics.sphrandn(jl.Float64, S, 2*S-1)

            
            self.P = jl.FastSphericalHarmonics.plan_sph2fourier(F)
            self.PA = jl.FastSphericalHarmonics.plan_sph_analysis(F)
            self.PS = jl.FastSphericalHarmonics.plan_sph_synthesis(F)


            self.jl = jl
            self.step2 = self.step2_fastTransforms
            self.step2_H = self.step2_H_fastTransforms


            n_phi = 2 * S - 1
            n_theta = S


            phi = 2 * np.pi * np.arange(n_phi) / n_phi
            theta = np.pi * (np.arange(n_theta) + 0.5) / n_theta  # uniform
            mu = jl.FastTransforms.chebyshevmoments1(jl.Float64, n_theta)
            self.weights = jl.FastTransforms.fejerweights1(mu)


        self.phi = phi
        self.theta = theta
        n_interp = n_radial
        if self.reduce_memory:
            n_interp = 2 * n_radial

        self.n_radial = n_radial
        self.n_phi = n_phi
        self.n_theta = n_theta
        self.n_interp = n_interp

        self.N = N

        mdmax = np.max(mds)
        self.mdmax = mdmax
        self.mds = mds

        # Make a list of lists: idx_list[l] is the index of all
        # sequential index values i with \ell value l, used for the interpolation
        idx_list = [None] * (lmax + 1)

        for i in range(lmax + 1):
            idx_list[i] = []
        for i in range(ne):
            l = ls[i]
            if ms[i] == 0:
                idx_list[l].append(i)

        self.idx_list = idx_list

        # self.idlm_list[l][m] contains all the sequential
        # indices i with physical indices l,m
        idlm_list = [
            [None for _ in range(2 * self.lmax + 1)]
            for _ in range(self.lmax + 1)
        ]
        for l in range(self.lmax + 1):
            for md in range(2 * self.lmax + 1):
                idlm_list[l][md] = []

        for i in range(ne):
            l = ls[i]
            md = mds[i]
            idlm_list[l][md].append(i)

        self.idlm_list = idlm_list


        ######## Create NUFFT plans
        lmd0 = np.min(lmds)
        lmd1 = np.max(lmds)

        if ne == 1:
            lmd1 = lmd1 * (1 + 2e-16)

        tmp_pts = 1 - (2 * np.arange(n_radial) + 1) / (2 * n_radial)
        tmp_pts = np.cos(np.pi * tmp_pts)
        pts = (tmp_pts + 1) / 2
        pts = (lmd1 - lmd0) * pts + lmd0

        pts = pts.reshape(-1, 1)

        R = N // 2
        h = 1 / R ** (1.5)

        self.R = R
        self.N = N
        self.h = h

        phi = phi.reshape(1, -1)
        theta = theta.reshape(-1, 1)

        x = np.cos(phi) * np.sin(theta)
        x = x.reshape(1, -1)
        y = np.sin(phi) * np.sin(theta)
        y = y.reshape(1, -1)
        z = np.ones(phi.shape) * np.cos(theta)
        z = z.reshape(1, -1)

        x = x * pts / R
        y = y * pts / R
        z = z * pts / R

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

        nufft_type = 2
        self.plan2 = finufft.Plan(
            nufft_type,
            (N, N, N),
            n_trans=1,
            eps=epsnufH,
            isign=-1,
            dtype=np.complex128,
        )
        self.plan2.setpts(y, x, z) # this ordering gives the ordering of the grid points in the paper

        nufft_type = 1
        self.plan1 = finufft.Plan(
            nufft_type,
            (N, N, N),
            n_trans=1,
            eps=epsnuf,
            isign=1,
            dtype=np.complex128,
        )
        self.plan1.setpts(y, x, z) # this ordering gives the ordering of the grid points in the paper

        # Source points for interpolation, i.e., Chebyshev nodes in the radial direction
        # The way we set up the interpolation below is with source and target radii
        # between 0 and 1, so we use xs and not the variable pts defined above.
        xs = 1 - (2 * np.arange(n_interp) + 1) / (2 * n_interp)
        xs = np.cos(np.pi * xs)

        if numsparse <= 0:
            ws = self.get_weights(xs)

        A3 = [None] * (lmax + 1)
        A3_T = [None] * (lmax + 1)

        b_sz = (n_interp, 2 * lmax + 1)
        b = np.zeros(b_sz)
        for i in range(lmax + 1):
            # Source function values
            ys = b[:, i]
            ys = ys.flatten()

            # Target points for interpolation, i.e., \lambda_{\ell k}
            # for all k that are included after truncation by lambda
            x = 2 * (lmds[idx_list[i]] - lmd0) / (lmd1 - lmd0) - 1

            _, x_ind, _ = np.intersect1d(x, xs, return_indices=True)
            x[x_ind] = x[x_ind] + 2e-16

            n = len(x)
            mm = len(xs)

            # if s is less than or equal to 0 we just do dense
            if numsparse > 0:
                A3[i], A3_T[i] = self.barycentric_interp_sparse(
                    x, xs, ys, numsparse
                )
            else:
                A3[i] = np.zeros((n, mm))
                denom = np.zeros(n)
                for j in range(mm):
                    xdiff = x - xs[j]
                    temp = ws[j] / xdiff
                    A3[i][:, j] = temp.flatten()
                    denom = denom + temp
                denom = denom.reshape(-1, 1)
                A3[i] = A3[i] / denom
                A3_T[i] = A3[i].T

        self.A3 = A3
        self.A3_T = A3_T

        # Set up indices indicating the complement of the unit ball
        xtmp = np.arange(-R, R + N % 2)
        ytmp = np.arange(-R, R + N % 2)
        ztmp = np.arange(-R, R + N % 2)
        xstmp, ystmp, zstmp = np.meshgrid(xtmp, ytmp, ztmp)
        xstmp = xstmp / R
        ystmp = ystmp / R
        zstmp = zstmp / R
        rstmp = np.sqrt(xstmp**2 + ystmp**2 + zstmp**2)
        idx = rstmp > 1 + 1e-13

        self.idx = idx

    def create_denseB(self, numthread=1):
        #####
        # NOTE THE FOLLOWING ISSUE WITH SPL.SPH_HARM:
        # https://github.com/scipy/scipy/issues/7778
        # We therefore use pyshtools for large m,
        # although pyshtools is slower. The cutoff m = 75
        # works for N <= 64. For larger N, make the
        # cutoff 75 smaller if you want to compute the dense matrix
        #####
        if self.N > 32:
            from pyshtools.expand import spharm_lm

        psi = [None] * self.ne
        for i in range(self.ne):
            l = self.ls[i]
            m = self.ms[i]
            lmd = self.lmds[i]
            c = self.cs[i]


            if (np.abs(m) <= 75) or (self.N <= 32):
                if m >= 0:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * spl.sph_harm(m, l, p, t)
                        * (r <= 1)
                    )
                else:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * (-1) ** int(m)
                        * np.conj(spl.sph_harm(np.abs(m), l, p, t))
                        * (r <= 1)
                    )

            else:
                if m >= 0:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * spharm_lm(
                            l,
                            m,
                            t,
                            p,
                            kind="complex",
                            degrees=False,
                            csphase=-1,
                            normalization="ortho",
                        )
                        * (r <= 1)
                    )
                else:
                    psi[i] = (
                        lambda r, t, p, c=c, l=l, m=m, lmd=lmd: c
                        * spl.spherical_jn(l, lmd * r)
                        * (-1) ** int(m)
                        * np.conj(
                            spharm_lm(
                                l,
                                np.abs(m),
                                t,
                                p,
                                kind="complex",
                                degrees=False,
                                csphase=-1,
                                normalization="ortho",
                            )
                        )
                        * (r <= 1)
                    )
        self.psi = psi

        # Evaluate eigenfunctions
        R = self.N // 2
        h = 1 / R ** (1.5)
        x = np.arange(-R, R + self.N % 2)
        y = np.arange(-R, R + self.N % 2)
        z = np.arange(-R, R + self.N % 2)
        xs, ys, zs = np.meshgrid(
            x, y, z
        ) 
        xs = xs / R
        ys = ys / R
        zs = zs / R
        rs = np.sqrt(xs**2 + ys**2 + zs**2)
        ps = np.arctan2(ys, xs)
        ps = ps + 2 * np.pi * (
            ps < 0
        )  # changes the phi definition interval from (-pi, pi) to (0,2*pi)
        ts = np.arctan2(np.sqrt(xs**2 + ys**2), zs)

        # Compute in parallel if numthread > 1
        from tqdm import tqdm
        from joblib import Parallel, delayed

        if numthread <= 1:
            B = np.zeros(
                (self.N, self.N, self.N, self.ne),
                dtype=np.complex128,
                order="F",
            )
            for i in tqdm(range(self.ne)):
                B[:, :, :, i] = self.psi[i](rs, ts, ps)
            B = h * B
        else:
            func = lambda i, rs=rs, ts=ts: self.psi[i](rs, ts, ps)
            B_list = Parallel(n_jobs=numthread, prefer="threads")(
                delayed(func)(i) for i in range(self.ne)
            )
            B_par = np.zeros(
                (self.N, self.N, self.N, self.ne),
                dtype=np.complex128,
                order="F",
            )
            for i in range(self.ne):
                B_par[:, :, :, i] = B_list[i]
            B = h * B_par

        if self.N > self.N1:
            B = B[: self.N1, : self.N1, : self.N1, :]
        B = B.reshape(self.N1**3, self.ne)

        if not self.complexmode:
            B = self.transform_complex_to_real(B, self.ms)

        return B.reshape(self.N1**3, self.ne)

    def lap_eig_ball(self, ne, bandlimit, max_l=None):
        # Computes dense matrix representation of the basis transform,
        # using the complex representation of the basis.

        # number of roots to check

        if not max_l:
            max_l = int(2.5 * ne ** (1 / 3))
        max_k = int(2.5 * ne ** (1 / 3))

        # preallocate
        ls = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        ks = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        ms = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        mds = np.zeros((max_l * (2 * max_l + 1) * max_k), dtype=int, order="F")
        cs = np.zeros(
            (max_l * (2 * max_l + 1) * max_k), dtype=np.float64, order="F"
        )
        lmds = (
            np.ones((max_l * (2 * max_l + 1) * max_k), dtype=np.float64)
            * np.inf
        )

        # load table of roots of jn (the scipy code has an issue where it gets
        # stuck in an infinite loop in Newton's method as of June 2022)
        path_to_module = os.path.dirname(__file__)
        zeros_path = os.path.join(path_to_module, "jl_zeros_l=3000_k=2500.mat")

        data = loadmat(zeros_path)
        roots_table = data["roots_table"]
        cs_path = os.path.join(path_to_module, "cs_l=3000_k=2500.mat")
        data = loadmat(cs_path)
        cs_table = data["cs"]
        # Sweep over lkm in the following order: k in {1,kmax}, l in {0,lmax}, m in {0,-1,1,-2, ...,-l,l}

        # If we notice that for a given k and l, the current root
        # is larger than the largest one in the list (and the list has length at least ne), then all other l for same k
        # and all other k for same l will be superfluous, so l will only have to up to this particular l - 1.

        ind = 0
        stop_l = max_l
        largest_lmd = 0
        for k in range(1, max_k):
            for l in range(stop_l):
                m_range = 2 * l + 1
                ks[ind : ind + m_range] = k
                ls[ind : ind + m_range] = l
                m_indices = np.arange(m_range)
                ms[ind : ind + m_range] = (-1) ** m_indices * (
                    (m_indices + 1) // 2
                )
                mds[ind : ind + m_range] = 2 * np.abs(
                    ms[ind : ind + m_range]
                ) - (ms[ind : ind + m_range] < 0)
                new_lmd = roots_table[l, k - 1]
                lmds[ind : ind + m_range] = new_lmd
                cs[ind : ind + m_range] = cs_table[l, k - 1]
                ind += m_range
                if (ind >= ne) and (new_lmd > largest_lmd):
                    stop_l = l
                    break
                largest_lmd = max(largest_lmd, new_lmd)




        idx = np.argsort(lmds[:ind], kind="stable")


        ls = ls[idx[:ne]]
        ks = ks[idx[:ne]]
        ms = ms[idx[:ne]]
        mds = mds[idx[:ne]]
        lmds = lmds[idx[:ne]]
        cs = cs[idx[:ne]]

        if bandlimit:
            threshold = (
                bandlimit * np.pi / 2
            )
            ne = np.searchsorted(lmds, threshold, side="left") - 1

        # potentially subtract 1 from ne to keep -m, +m pairs
        if ms[ne - 1] < 0:
            ne = ne - 1

        # make sure that ne is always at least 1
        if ne <= 1:
            ne = 1

        # # take top ne values (with the new ne)
        ls = ls[:ne]
        ks = ks[:ne]
        ms = ms[:ne]
        lmds = lmds[:ne]
        mds = mds[:ne]
        cs = cs[:ne]


        return ls, ks, ms, mds, lmds, cs, ne

    def precomp_transform_complex_to_real(self, ms):
        ne = len(ms)
        nnz = np.sum(ms == 0) + 2 * np.sum(ms != 0)
        idx = np.zeros(nnz, dtype=int)
        jdx = np.zeros(nnz, dtype=int)
        vals = np.zeros(nnz, dtype=np.complex128)

        k = 0
        for i in range(ne):
            m = ms[i]
            if m == 0:
                vals[k] = 1
                idx[k] = i
                jdx[k] = i
                k = k + 1
            if m < 0:
                s = (-1) ** np.abs(m)

                vals[k] = -1j / np.sqrt(2)
                idx[k] = i
                jdx[k] = i
                k = k + 1

                vals[k] = s * 1j / np.sqrt(2)
                idx[k] = i
                jdx[k] = i + 1
                k = k + 1

                vals[k] = 1 / np.sqrt(2)
                idx[k] = i + 1
                jdx[k] = i
                k = k + 1

                vals[k] = s / (np.sqrt(2))
                idx[k] = i + 1
                jdx[k] = i + 1
                k = k + 1

        A = spr.csr_matrix(
            (vals, (idx, jdx)), shape=(ne, ne), dtype=np.complex128
        )
        return A

    def transform_complex_to_real(self, Z, ms):
        ne = Z.shape[1]
        X = np.zeros(Z.shape, dtype=np.float64)

        for i in range(ne):
            m = ms[i]
            if m == 0:
                X[:, i] = np.real(Z[:, i])
            if m < 0:
                s = (-1) ** np.abs(m)
                x0 = (Z[:, i] - s * Z[:, i + 1]) * 1j / np.sqrt(2)
                x1 = (Z[:, i] + s * Z[:, i + 1]) / (np.sqrt(2))
                X[:, i] = np.real(x0)
                X[:, i + 1] = np.real(x1)

        return X

    def transform_real_to_complex(self, X, ms):
        ne = X.shape[1]
        Z = np.zeros(X.shape, dtype=np.complex128)

        for i in range(ne):
            m = ms[i]
            if m == 0:
                Z[:, i] = X[:, i]
            if m < 0:
                s = (-1) ** np.abs(m)
                z0 = (-1j * X[:, i] + X[:, i + 1]) / np.sqrt(2)
                z1 = s * (X[:, i] + 1j * X[:, i + 1]) / np.sqrt(2)
                Z[:, i] = z0
                Z[:, i + 1] = z1

        return Z

    def get_weights(self, xs):
        m = len(xs)
        I = np.ones(m, dtype=bool)
        I[0] = False
        e = np.sum(-np.log(np.abs(xs[0] - xs[I])))
        const = np.exp(e / m)
        ws = np.zeros(m)
        I = np.ones(m, dtype=bool)
        for j in range(m):
            I[j] = False
            xt = const * (xs[j] - xs[I])
            ws[j] = 1 / np.prod(xt)
            I[j] = True

        return ws

    def evaluate(self, a):

        if not self.complexmode:
            a = self.r2c @ a.flatten()

        f = self.step1_H(self.step2_H(self.step3_H(a)))
        f = f.reshape(self.N, self.N, self.N)

        if self.N > self.N1:
            f = f[: self.N1, : self.N1, : self.N1]
        return f

    def evaluate_t(self, f):
        f = np.copy(f).reshape(self.N1, self.N1, self.N1)


        if self.N > self.N1:
            f = np.pad(f, ((0, 1), (0, 1), (0, 1)))

        # Remove pixels outside disk
        f[self.idx] = 0
        f = f.flatten()


        a = self.step3(self.step2(self.step1(f))) * self.h
        if not self.complexmode:
            a = self.c2r @ a.flatten()

        return a

   

    def step1(self, f):

        f = f.reshape(self.N, self.N, self.N)
        f = np.array(f, dtype=np.complex128)

        z = np.zeros(
            (self.n_radial, self.n_theta, self.n_phi), dtype=np.complex128
        )
        z0 = self.plan2.execute(f)

        if self.sph_harm_solver == "FastTransforms.jl":
            z = z0.reshape(self.n_radial, self.n_theta, self.n_phi)
        else:
            if self.force_real:
                z0 = z0.reshape(self.n_radial, self.n_theta, self.n_phi // 2)
                z[:, :, : self.n_phi // 2] = z0
                z[:, ::-1, self.n_phi // 2 :] = np.conj(z0)
            else:
                z = z0.reshape(self.n_radial, self.n_theta, self.n_phi)


        z = z.flatten()

        return z




    def step2_torch(self, z):
        # https://github.com/NVIDIA/torch-harmonics

        z = z.reshape(self.n_radial, self.n_theta, self.n_phi)
        # From https://arxiv.org/pdf/1202.6522.pdf, bottom of page 3, torch only
        # computes the coefficients for real-valued data and then only for m >= 0.

        breal = self.torch_reshape_order_t(
            self.sht(self.torch.DoubleTensor(np.real(z)).to(self.device))
            .cpu()
            .numpy()
        )
        bimag = self.torch_reshape_order_t(
            self.sht(self.torch.DoubleTensor(np.imag(z)).to(self.device))
            .cpu()
            .numpy()
        )

        b = breal + 1j * bimag

        for l in range(self.lmax + 1):
            b[:, l, :] = b[:, l, :] * (1j) ** l / (4 * np.pi)

        return b

    def step2_fastTransforms(self, z):

        z = z.reshape(self.n_radial, self.n_theta, self.n_phi)
        b = np.zeros(
            (self.n_radial, self.lmax + 1, 2 * self.lmax + 1),
            dtype=np.complex128,
        )
        for q in range(self.n_radial):
            Fr = self.jl.Matrix(np.real(z[q, :, :]))
            Fi = self.jl.Matrix(np.imag(z[q, :, :]))

            Gr = self.PA*Fr
            tmpr = np.complex128(self.jl.ldiv(self.P, Gr))

            Gi = self.PA*Fi
            tmpi = np.complex128(self.jl.ldiv(self.P, Gi))

            b[q, :, :] = self.fastTransforms_reshape_order_t(
                tmpr
            ) + 1j * self.fastTransforms_reshape_order_t(
                tmpi
            )

        for l in range(self.lmax + 1):
            b[:, l, :] = b[:, l, :] * (1j) ** l / (4 * np.pi)

        return b





    def torch_reshape_order_t(self, b):
        # converts the order of m returned by torch to 0,-1,1,-2,2,-3,3,...
        # torch only computes the coefficients for m >= 0, but can use
        # alpha_{l,-m} = (-1)**m*\overline{alpha_{l,m}} for real-valued structures.
        # We therefore separate the input to torch_step2 into real and imaginary parts.


        s = b.shape
        bn = np.zeros((s[0], s[1], 2 * s[2] - 1), dtype=np.complex128)

        bn[:, :, 0] = b[:, :, 0]

        bn[:, :, 1 : (2 * s[2] - 1) : 4] = np.conj(b[:, :, 1 : s[2] : 2]) * (-1)
        bn[:, :, 3 : (2 * s[2] - 1) : 4] = np.conj(b[:, :, 2 : s[2] : 2])
        bn[:, :, 2 : (2 * s[2]) : 2] = b[:, :, 1 : s[2]]

        return bn

    def fastTransforms_reshape_order_t(self, b):
        # converts the order of m returned by FastTransforms.jl
        # to have columns sweeping the m-index as 0,-1,1,-2,2,-3,3,...
        bn = np.zeros((self.lmax + 1, 2 * self.lmax + 1), dtype=np.complex128)
        for l in range(self.lmax + 1):
            for m in range(l + 1):
                indpos = self.jl.sph_mode(l, m)
                indneg = self.jl.sph_mode(l, -m)
                # julia has 1-indexing and the package does real-valued harmonics
                # The convention below is different from ordinary conventions for
                # real-valued harmonics, but seems to be what they are using.
                if m > 0:
                    bn[l, 2 * m] = (
                        (-1) ** m
                        / np.sqrt(2)
                        * (
                            b[indpos[1] - 1, indpos[2] - 1]
                            - 1j * b[indneg[1] - 1, indneg[2] - 1]
                        )
                    )
                    bn[l, 2 * m - 1] = (
                        1
                        / np.sqrt(2)
                        * (
                            b[indpos[1] - 1, indpos[2] - 1]
                            + 1j * b[indneg[1] - 1, indneg[2] - 1]
                        )
                    )

                else:
                    bn[l, m] = b[indpos[1] - 1, indpos[2] - 1]
        return bn

    def step3(self, b):
        if self.n_interp > self.n_radial:
            b = dct(b, axis=0, type=2) / (2 * self.n_radial)
            bz = np.zeros(b.shape)
            b = np.concatenate((b, bz), axis=0)
            b = idct(b, axis=0, type=2) * 2 * b.shape[0]

        #the below is a faster version of
        # a = np.zeros(self.ne, dtype=np.complex128)
        # for i in range(self.ne):
        #     l = self.ls[i]
        #     md = self.mds[i]
        #     a[self.idlm_list[l][md]] = (
        #         (self.A3[l] @ b[:, l, md])[: len(self.idlm_list[l][md])]
        #     )
        # a = a * self.cs
        # a = a.flatten()

        a = np.zeros(self.ne, dtype=np.complex128)
        for l in range(self.lmax + 1):
            tmp = self.A3[l] @ b[:, l, :]

            m_range = 2 * l + 1

            inds = np.concatenate(
                [
                    self.idlm_list[l][md]
                    for md in range(m_range)
                    if self.idlm_list[l][md]
                ]
            )
            rhs = np.concatenate(
                [tmp[: len(self.idlm_list[l][md]), md] for md in range(m_range)]
            )

            a[inds] = rhs

        a = a * self.cs
        a = a.flatten()

        return a

    def step1_H(self, z):

        if self.sph_harm_solver == "FastTransforms.jl":
            # Whole z
            f = self.plan1.execute(z.flatten())
            f = f.reshape(self.N, self.N, self.N)
            f[self.idx] = 0
            f = f.flatten()
        else:
            if self.force_real:
                # Half z
                z = z[:, :, : self.n_phi // 2]
                f = self.plan1.execute(z.flatten())
                f = 2 * np.real(f)
                f = f.reshape(self.N, self.N, self.N)
                f[self.idx] = 0
                f = f.flatten()
            else:
                # Whole z
                f = self.plan1.execute(z.flatten())
                f = f.reshape(self.N, self.N, self.N)
                f[self.idx] = 0
                f = f.flatten()
                

        return f

    def step2_H_torch(self, b):
        for l in range(self.lmax + 1):
            b[:, l, :] = (
                (-1j) ** l / (4 * np.pi) * b[:, l, :] * 2 * np.pi / self.n_phi
            )

        b1, b2 = self.torch_reshape_order(b)
        b1 = self.torch.tensor(b1, dtype=self.torch.complex128)
        b2 = self.torch.tensor(b2, dtype=self.torch.complex128)

        z1 = np.complex128(self.isht(b1.to(self.device)).cpu().numpy())
        z2 = np.complex128(self.isht(b2.to(self.device)).cpu().numpy())
        z = z1 + 1j * z2

        for i in range(len(self.weights)):
            z[:, i, :] = z[:, i, :] * np.conj(self.weights[i])

        return z




    def step2_H_fastTransforms(self, b):

        for l in range(self.lmax + 1):
            b[:, l, :] = (
                (-1j) ** l / (4 * np.pi) * b[:, l, :] * 2 * np.pi / self.n_phi
            )

        z = np.zeros(
            (self.n_radial, self.n_theta, self.n_phi), dtype=np.complex128
        )
        bq = self.fastTransforms_reshape_order(b)

        for q in range(self.n_radial):
            b1 = bq[q, :, :]
            b1 = self.jl.Matrix(b1)
            G=self.P*b1
            H = self.PS*G
            # z[q, :, :] = np.complex128(self.jl.sph_evaluate(self.jl.Matrix(b1)))
            z[q, :, :] = np.complex128(H)

        for i in range(len(self.weights)):
            z[:, i, :] = z[:, i, :] * np.conj(self.weights[i])

        return z

    def my_pad(self,v,lv,k):
        if k == 0:
            return v
        w = np.zeros((lv+k,),dtype=np.complex128)
        w[:lv] = v
        return w
    
    def step3_H(self, a):

 
        a = a * self.h
        a = a.flatten()
        a = a * self.cs

 
        b = np.zeros(
            (self.n_interp, self.lmax + 1, 2 * self.lmax + 1),
            dtype=np.complex128,
            order="F",
        )
        ## the below is required for the vectorization trick in the loop
        a = np.conj(a)
        for l in range(self.lmax + 1):
            m_range = 2 * l + 1

            #the below is a faster version of
            # for md in range(m_range):
            #     b[:, l, md] = (
            #         np.conj(self.A3_T[l][:, : len(self.idlm_list[l][md])])
            #         @ a[self.idlm_list[l][md]]
            #     )

            tli = self.idlm_list[l]
            ts = self.A3_T[l].shape[1]

            tmp = np.concatenate([self.my_pad(a[tli[md]],len(tli[md]), ts-len(tli[md])) for md in range(m_range)]).reshape(-1,m_range,order='F')
            b[:, l, :m_range] = np.conj(self.A3_T[l]@tmp)    

        if self.n_interp > self.n_radial:
            b = dct(b, axis=0, type=2)
            b = b[: self.n_radial, :]
            b = idct(b, axis=0, type=2)

        return b

    def torch_reshape_order(self, b):
        # converts the column order of b from 0,-1,1,-2,2,-3,3,...
        # to the order required by torch.
        # torch only computes the coefficients for m >= 0, but can use
        # alpha_{l,-m} = (-1)**m*\overline{alpha_{l,m}} for real-valued structures.
        # We therefore separated the input to torch_step2 into real and imaginary parts,
        # and now need to invert this separation

        s = b.shape
        tmp1 = b[:, :, 0 : s[2] : 2]
        # Every other one here has to have their sign flipped
        signs = [(-1) ** (k + 1) for k in range(self.lmax)]
        tmp2 = np.concatenate(
            (b[:, :, 0].reshape(s[0], s[1], 1), b[:, :, 1 : s[2] : 2] * signs),
            axis=2,
        )
        b1 = 0.5 * np.real(tmp1 + tmp2) + 1j * 0.5 * np.imag(tmp1 - tmp2)
        b2 = 0.5 * np.imag(tmp1 + tmp2) - 1j * 0.5 * np.real(tmp1 - tmp2)

        return b1, b2

    def fastTransforms_reshape_order(self, b):
        # converts the order with columns sweeping the m-index as 0,-1,1,-2,2,-3,3,...
        # to that required by FastTransforms.jl
        s = b.shape
        bn = np.zeros((s[0], self.n_theta, self.n_phi), dtype=np.complex128)
        for l in range(self.lmax + 1):
            for m in range(l + 1):
                indpos = self.jl.sph_mode(l, m)
                indneg = self.jl.sph_mode(l, -m)
                # julia has 1-indexing and the package does real-valued harmonics
                # The convention below is different from ordinary conventions for
                # real-valued harmonics, but seems to be what they are using.
                if m > 0:
                    bn[:, indpos[1] - 1, indpos[2] - 1] = (
                        1
                        / np.sqrt(2)
                        * ((-1) ** m * b[:, l, 2 * m] + b[:, l, 2 * m - 1])
                    )
                    bn[:, indneg[1] - 1, indneg[2] - 1] = (
                        1
                        / np.sqrt(2)
                        * (
                            1j * (-1) ** m * b[:, l, 2 * m]
                            - 1j * b[:, l, 2 * m - 1]
                        )
                    )

                else:
                    bn[:, l, m] = b[:, l, m]
        return bn
    
    
    def expand(self, f, toltype='l1linf'):
        b = self.evaluate_t(f)
        a0 = self.expand_alpha*b
        if toltype == 'l1linf':
            no = np.linalg.norm(a0,np.Inf)
            n1 = 1
        elif toltype == 'l2':
            no = np.linalg.norm(a0,2)
            n1 = 2
        for iter in range(self.maxitr):
            a0old = a0
            a0 = a0 - self.expand_alpha*(self.evaluate_t(self.evaluate(a0))) + self.expand_alpha*b
            if np.linalg.norm(a0-a0old,n1)/no < self.expand_rel_tol:
                break
        return a0

    def lowpass(self, a, bandlimit):
        threshold = (
            bandlimit * np.pi / 2
        )  
        ne = np.searchsorted(self.lmds, threshold, side="left") - 1
        a[ne::] = 0
        return a
    


    def barycentric_interp_sparse(self, x, xs, ys, s):
        # https://people.maths.ox.ac.uk/trefethen/barycentric.pdf

        n = len(x)
        m = len(xs)

        # Modify points by 2e-16 to avoid division by zero
        vals, x_ind, xs_ind = np.intersect1d(
            x, xs, return_indices=True, assume_unique=True
        )
        x[x_ind] = x[x_ind] + 2e-16

        idx = np.zeros((n, s))
        jdx = np.zeros((n, s))
        vals = np.zeros((n, s))
        xss = np.zeros((n, s))
        idps = np.zeros((n, s))
        numer = np.zeros((n, 1))
        denom = np.zeros((n, 1))
        temp = np.zeros((n, 1))
        ws = np.zeros((n, s))
        xdiff = np.zeros(n)
        for i in range(n):
            # get a kind of balanced interval around our point
            k = np.searchsorted(x[i] < xs, True)

            idp = np.arange(k - s // 2, k + (s + 1) // 2)
            if idp[0] < 0:
                idp = np.arange(s)
            if idp[-1] >= m:
                idp = np.arange(m - s, m)
            xss[i, :] = xs[idp]
            jdx[i, :] = idp
            idx[i, :] = i

        x = x.reshape(-1, 1)
        Iw = np.ones(s, dtype=bool)
        ew = np.zeros((n, 1))
        xtw = np.zeros((n, s - 1))

        Iw[0] = False
        const = np.zeros((n, 1))
        for j in range(s):
            ew = np.sum(
                -np.log(np.abs(xss[:, 0].reshape(-1, 1) - xss[:, Iw])), axis=1
            )
            constw = np.exp(ew / s)
            constw = constw.reshape(-1, 1)
            const += constw
        const = const / s

        for j in range(s):
            Iw[j] = False
            xtw = const * (xss[:, j].reshape(-1, 1) - xss[:, Iw])
            ws[:, j] = 1 / np.prod(xtw, axis=1)
            Iw[j] = True

        xdiff = xdiff.flatten()
        x = x.flatten()
        temp = temp.flatten()
        denom = denom.flatten()
        for j in range(s):
            xdiff = x - xss[:, j]
            temp = ws[:, j] / xdiff
            vals[:, j] = vals[:, j] + temp
            denom = denom + temp
        vals = vals / denom.reshape(-1, 1)

        vals = vals.flatten()
        idx = idx.flatten()
        jdx = jdx.flatten()
        A = spr.csr_matrix((vals, (idx, jdx)), shape=(n, m), dtype=np.float64)
        A_T = spr.csr_matrix((vals, (jdx, idx)), shape=(m, n), dtype=np.float64)

        return A, A_T
