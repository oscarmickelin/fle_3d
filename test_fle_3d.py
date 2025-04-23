import time
from os.path import exists
from scipy.io import savemat
from fle_3d import FLEBasis3D
import numpy as np
from scipy.io import loadmat
import mrcfile
import matplotlib.pyplot as plt


def main():
    #######
    # If True, reduces the number of radial points in defining
    # NUFFT grids, and does an alternative interpolation to
    # compensate. To reproduce the tables and figures of the
    # paper, set this to False. 
    reduce_memory = True
    #######


    # Test 0: Run several quick test on volumes of size 7 and 8
    print("test 0") 
    test0_quick_tests()

    # # test 1: Verify that code agrees with dense matrix mulitplication
    print("test 1")
    print('... testing nvidia_torch and real mode')
    test1_fle_vs_dense("nvidia_torch", "real", reduce_memory)
    print('... testing nvidia_torch and complex mode')
    test1_fle_vs_dense("nvidia_torch", "complex", reduce_memory)

    # ##########################
    # # NOTE: there are compatibility issues when loading both
    # # torch and julia, in a single python shell.
    # # If the following code crashes, comment out the torch tests
    # # above and uncomment the julia tests below,
    # # and run them in a new shell.
    # print('... testing FastTransforms.jl and real mode')
    # test1_fle_vs_dense("FastTransforms.jl", "real", reduce_memory)
    # print('... testing FastTransforms.jl and complex mode')
    # test1_fle_vs_dense("FastTransforms.jl", "complex", reduce_memory)
    # ##########################

    # # test 2: verify that code can lowpass
    #print("test 2")
    #test2_fle_lowpass(reduce_memory)

    ## test 3: verify timing 
    #print("test 3")
    #test3_part_timing(reduce_memory)

    # ## # test 4: check the error of
    # ## least-squares expansions into the basis
    # print("test 4")
    # test4_expand_error_test(reduce_memory)

    # print("test 5")
    # test5(32)

    print("test 6")
    test6_visualize_eigenfunctions_for_odd_even_N()

    return

def test0_quick_tests():

    sph_harm_solver = "nvidia_torch"
    mode = "real"
    reduce_memory = False


    Ns = [7,8]
    eps = 1e-14
    
    n = len(Ns)
    erra = np.zeros(n)
    errx = np.zeros(n)
    erra2 = np.zeros(n)
    errx2 = np.zeros(n)
    
    print("eps =",eps)
    for i,N in enumerate(Ns):
        print("N =",N)
        bandlimit = N
        fle = FLEBasis3D(N, bandlimit, eps, sph_harm_solver=sph_harm_solver,mode=mode,reduce_memory=reduce_memory)
        B = fle.create_denseB(numthread=1)
        erra, errx, erra2, errx2 = test0_fle_vs_dense_helper(sph_harm_solver,mode,N, eps, B, reduce_memory)
        print(erra)
        print(errx)
        print(erra2)
        print(errx2)
        print("")

        print("Test timing code runs")
        x = np.random.rand(N,N,N) 
        x = x / np.max(np.abs(x.flatten()))
        dts = test3_helper(N, fle, x)
        print(dts)
        print("")

    
        print("Test expand code runs")
        bandlimit = N
        fle = FLEBasis3D(N, bandlimit, eps, sph_harm_solver=sph_harm_solver,expand_eps=eps, mode="complex", reduce_memory=reduce_memory)

        a0 = fle.expand(x)
        x0 = fle.evaluate(a0)

        err = np.linalg.norm(x.flatten() - x0.flatten(),np.inf)/np.linalg.norm(x.flatten(), 1)
        err2 = np.linalg.norm(x.flatten() - x0.flatten(),2)/np.linalg.norm(x.flatten(), 2)
        print(err)
        print(err2)
    return



def test0_fle_vs_dense_helper(sph_harm_solver,mode,N, eps, B, reduce_memory):

    # Parameters
    # Basis pre-computation
    bandlimit = N
    fle = FLEBasis3D(N, bandlimit, eps, sph_harm_solver=sph_harm_solver,mode=mode,reduce_memory=reduce_memory)


    x = np.random.rand(N,N,N) 
    x = x / np.max(np.abs(x.flatten()))


    # evaluate_t
    a_dense = np.conj(B.T) @ (x.flatten())
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # print
    errx = np.linalg.norm(a_dense - a_fle,np.inf)/np.linalg.norm(x.flatten(), 1)
    erra = np.linalg.norm(xlow_dense - xlow_fle.flatten(),np.inf)/np.linalg.norm(a_dense, 1)

    errx2 = np.linalg.norm(a_dense - a_fle,2)/np.linalg.norm(a_dense.flatten(), 2)
    erra2 = np.linalg.norm(xlow_dense - xlow_fle.flatten(),2)/np.linalg.norm(xlow_dense, 2)
    return erra, errx, erra2, errx2





def test1_fle_vs_dense(sph_harm_solver,mode, reduce_memory):

    Ns = [33]
    Nns = []
    epss = []
    
    for N in Ns:        
    	for eps in (1e-4, 1e-7, 1e-10, 1e-14):
            Nns.append(N)
            epss.append(eps)
    n = len(Nns)
    erra = np.zeros(n)
    errx = np.zeros(n)
    erra2 = np.zeros(n)
    errx2 = np.zeros(n)
    
    i = 0
    for N in Ns:
        print('Precomputing FLE...')
        bandlimit = N
        fle = FLEBasis3D(N, bandlimit, 1e-4, sph_harm_solver=sph_harm_solver,mode=mode,reduce_memory=reduce_memory)
        print('Creating dense matrix...')
        B = fle.create_denseB(numthread=1)
        print('... dense matrix created')
        for eps in (1e-4, 1e-7, 1e-10, 1e-14):
            tmperra, tmperrx, tmperra2, tmperrx2 = test1_fle_vs_dense_helper(sph_harm_solver,mode,N, eps, B, reduce_memory)
            erra[i] = tmperra
            errx[i] = tmperrx
            erra2[i] = tmperra2
            errx2[i] = tmperrx2
            i += 1


    # make {tab:accuracy}
    print(sph_harm_solver,mode)
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$N$ & $\\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$& $\\text{l2 err}_a$ & $\\text{l2 err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Nns[i],
            "&",
            "{:12.5e}".format(epss[i]),
            "&",
            "{:12.5e}".format(erra[i]),
            "&",
            "{:12.5e}".format(errx[i]),
            "&",
            "{:12.5e}".format(erra2[i]),
            "&",
            "{:12.5e}".format(errx2[i]),
            "\\\\",
        )
        if i % len(Ns) == len(Ns) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


def test1_fle_vs_dense_helper(sph_harm_solver,mode,N, eps, B, reduce_memory):

    # Parameters
    # Bandlimit scaled so that N is maximum suggested bandlimit
    print('Running test with N='+ str(N))
    # Basis pre-computation
    bandlimit = N
    fle = FLEBasis3D(N, bandlimit, eps, sph_harm_solver=sph_harm_solver,mode=mode,reduce_memory=reduce_memory)


    # load example volume
    datafile = "test_volumes/data_N=" + str(N) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))


    # evaluate_t
    a_dense = np.conj(B.T) @ (x.flatten())
    a_fle = fle.evaluate_t(x)

    # evaluate
    xlow_dense = B @ a_dense
    xlow_fle = fle.evaluate(a_dense)

    # print
    errx = np.linalg.norm(a_dense - a_fle,np.inf)/np.linalg.norm(x.flatten(), 1)
    erra = np.linalg.norm(xlow_dense - xlow_fle.flatten(),np.inf)/np.linalg.norm(a_dense, 1)

    errx2 = np.linalg.norm(a_dense - a_fle,2)/np.linalg.norm(a_dense.flatten(), 2)
    erra2 = np.linalg.norm(xlow_dense - xlow_fle.flatten(),2)/np.linalg.norm(xlow_dense, 2)
    return erra, errx, erra2, errx2




def test2_fle_lowpass(reduce_memory):

    # Parameters
    # Use N x N x N volumes
    N = 128
    # Bandlimit scaled so that N is maximum suggested bandlimit
    bandlimit = N
    # Relative error compared to dense matrix method
    eps = 1e-6

    # Basis pre-computation
    fle = FLEBasis3D(N, bandlimit, eps, reduce_memory=reduce_memory)

    # load example volume
    datafile = "test_volumes/data_N=" + str(N) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))

    with mrcfile.new("FLEBasis3D N=" + str(N) + "original.mrc", overwrite=True) as mrc:
            mrc.set_data(np.float32(np.real(np.squeeze(np.float32(x))))) 

    # basic low pass
    for k in range(4):
        print("Dividing bandlimit by", 2**k)
        bandlimit = N // (2**k)
        a_fle = fle.expand(x)
        a_low = fle.lowpass(a_fle, bandlimit)
        xlow = fle.evaluate(a_low)
        print("bandlimit", bandlimit)
        print("num nonzero coeff", np.sum(a_low != 0))

        with mrcfile.new("FLEBasis3D N=" + str(N) + "bandlimit=" + str(bandlimit)+".mrc", overwrite=True) as mrc:
            mrc.set_data(np.float32(np.real(np.squeeze(np.float32(xlow)))))


def test3_part_timing(reduce_memory):

    nr = 1  # number of trials
    #Ns = [33,49,65,129]#,256]
    Ns = [32,33]
    eps = 1e-7
    n = len(Ns)

    dts = np.zeros((n, 6, nr))
    dts_dense = np.zeros((n, 2, nr))
    precomp = np.zeros((n,1))
    precomp_dense = np.zeros((n,1))
    for i in range(n):
        N = Ns[i]
        print('Running N =', N)
        bandlimit = N
        t1 = time.time()
        fle = FLEBasis3D(N, bandlimit, eps, reduce_memory=reduce_memory)
        dt = time.time() - t1
        precomp[i] = dt
        # load example volume
        datafile = "test_volumes/data_N=" + str(N) + ".mat"
        data = loadmat(datafile)
        x = data["x"]
        x = x / np.max(np.abs(x.flatten()))

        for j in range(nr):
            dts[i, :, j] = test3_helper(N, fle, x)

            if N <= 32:
                t1 = time.time()
                B = fle.create_denseB(numthread=1)
                BH = np.conj(B.T)
                dt = time.time() - t1
                ###### THE BELOW UNDERESTIMATES THE PRECOMPUTATION FOR THE DENSE ONE...
                ###### IT DOESN'T DO THE SETUP OF THE lambdas ETC.
                ###### THIS IS OKAY AND WE CAN MAYBE BE GRACIOUS AND LET IT DO THIS
                precomp_dense[i] = dt 
            
            
                t1 = time.time()
                a=BH@x.flatten() 
                dt = time.time() - t1
                dts_dense[i,0,j] = dt
                t1 = time.time()
                y=B@a.flatten() 
                dt = time.time() - t1
                dts_dense[i,1,j] = dt

    dts_avg = np.sum(dts, axis=2) / nr
    dts_dense_avg = np.sum(dts_dense, axis=2) / nr


    dts_avg = np.sum(dts, axis=2) / nr
    dts_dense_avg = np.sum(dts_dense, axis=2) / nr

    # make {tab:timing}
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & $dt_1$ & $dt_2$ & dt_3 & dt_1^H & dt_2^h & dt_3^h \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Ns[i],
            "&",
            "{:10.3e}".format(dts_avg[i, 0]),
            "&",
            "{:10.3e}".format(dts_avg[i, 1]),
            "&",
            "{:10.3e}".format(dts_avg[i, 2]),
            "&",
            "{:10.3e}".format(dts_avg[i, 3]),
            "&",
            "{:10.3e}".format(dts_avg[i, 4]),
            "&",
            "{:10.3e}".format(dts_avg[i, 5]),
            "\\\\",
        )
        if i % len(Ns) == len(Ns) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")

    print("precomputation:")
    print(r"\begin{tabular}{r|c}")
    print("$N$ & precomp time  \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Ns[i],
            "&",
            "{:10.3e}".format(precomp[i][0]),
            "\\\\",
        )
        if i % len(Ns) == len(Ns) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")

    print("dense precomputation:")
    print(r"\begin{tabular}{r|c}")
    print("$N$ & dense precomp time  \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Ns[i],
            "&",
            "{:10.3e}".format(precomp_dense[i][0]),
            "\\\\",
        )
        if i % len(Ns) == len(Ns) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")


    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$l$ & dense B^H & dense B \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            Ns[i],
            "&",
            "{:10.3e}".format(dts_dense_avg[i, 0]),
            "&",
            "{:10.3e}".format(dts_dense_avg[i, 1]),
            "\\\\",
        )
        if i % len(Ns) == len(Ns) - 1:
            print(r"\hline")
    print(r"\end{tabular}")
    print("")



def test3_helper(N, fle, x):

    # Get ready for Step 1
    N = fle.N
    f = np.copy(x).reshape(fle.N, fle.N, fle.N)
    f[fle.idx] = 0


    if fle.N2 > fle.N:
        f = np.pad(f, ((1, 0), (1, 0), (1, 0)))
    f = f.flatten()

    # Step 1
    t0 = time.time()
    z = fle.step1(f)
    t1 = time.time()
    dt1 = t1 - t0

    # Step 2
    t0 = time.time()
    b = fle.step2(z)
    t1 = time.time()
    dt2 = t1 - t0

    # Step 3
    t0 = time.time()
    a = fle.step3(b)
    t1 = time.time()
    dt3 = t1 - t0

    # Step 3 adjoint
    t0 = time.time()
    b = fle.step3_H(a)
    t1 = time.time()
    dt1H = t1 - t0

    # Step 2 adjoint
    t0 = time.time()
    z = fle.step2_H(b)
    t1 = time.time()
    dt2H = t1 - t0

    # Step 1 adjoint
    t0 = time.time()
    f = fle.step1_H(z)
    t1 = time.time()
    dt3H = t1 - t0


    dts = [dt1, dt2, dt3, dt1H, dt2H, dt3H]
    return dts




def test4_expand_error_test(reduce_memory):
    Nns = []
    epss = []
    for eps in [1e-4, 1e-7]:#, 1e-10, 1e-14]:
        for N in [33,65]:#[32,48,56,64,128,256]:
            Nns.append(N)
            epss.append(eps)
    n = len(Nns)
    err = np.zeros(n)
    err2 = np.zeros(n)
    for i in range(n):
        err[i], err2[i] = test4_helper(Nns[i], epss[i], reduce_memory)

    # make {tab:accuracy}
    print("expand test")
    for i in range(n):
        print(
            Nns[i],
            " ",
            "{:12.5e}".format(epss[i]),
            " ",
            "{:12.5e}".format(err[i]),
            " ",
            "{:12.5e}".format(err2[i]),
        )


def test4_helper(N, eps, reduce_memory):

    # Parameters
    # Bandlimit scaled so that N is maximum suggested bandlimit

    # Basis pre-computation
    print('Running N =',N)
    # bandlimit = int(1.25*N)
    bandlimit = N
    fle = FLEBasis3D(N, bandlimit, eps, expand_eps=eps, mode="complex", reduce_memory=reduce_memory)

    # load example volume
    datafile = "test_volumes/data_N=" + str(N) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = fle.evaluate(fle.evaluate_t(x))

    # evaluate_t
    a0 = fle.expand(x)
    x0 = fle.evaluate(a0)

    err = np.linalg.norm(x.flatten() - x0.flatten(),np.inf)/np.linalg.norm(x.flatten(), 1)
    err2 = np.linalg.norm(x.flatten() - x0.flatten(),2)/np.linalg.norm(x.flatten(), 2)

    return err, err2



def test5(N):

    # Parameters
    # Bandlimit scaled so that N is maximum suggested bandlimit

    # Basis pre-computation
    print('Running N =',N)
    # bandlimit = int(1.25*N)
    bandlimit = N
    eps = 1e-9
    fle = FLEBasis3D(N, bandlimit, eps, expand_eps=eps, mode="real", reduce_memory=True, maxfun = 30)

    # load example volume
    datafile = "test_volumes/data_N=" + str(N) + ".mat"
    data = loadmat(datafile)
    x = data["x"]
    x = x / np.max(np.abs(x.flatten()))
    x = fle.evaluate(fle.evaluate_t(x))

    B = fle.create_denseB()
    for ind in range(20):
        v = np.zeros((fle.ne,1), dtype = np.complex128)
        v[ind] = 1
        psi0 = fle.evaluate(v).flatten()
        psi = B[:,ind]

        x = np.linspace(-1,1,N,endpoint=False)
        xs, ys, zs = np.meshgrid(x, x, x) 

        psi2 = np.zeros(psi.shape, dtype = np.complex128)
        l = fle.ls[ind]
        m = fle.ms[ind]
        k = fle.ks[ind]


        rs = np.sqrt(xs**2 + ys**2 + zs**2)
        ps = np.arctan2(ys, xs)
        ts = np.arctan2(np.sqrt(xs**2 + ys**2), zs)
        import scipy.special as spl

        if m >= 0:
            tmpp = fle.cs[ind]*spl.spherical_jn(l, fle.lmds[ind] * rs)*spl.sph_harm(m, l, ps, ts)*( rs <= 1 )
            tmpm = fle.cs[ind]*spl.spherical_jn(l, fle.lmds[ind] * rs)*spl.sph_harm(-m, l, ps, ts)*( rs <= 1 )
            psi2 = (tmpm + (-1)**m*tmpp) *fle.h/2
        else:
            tmpp = fle.cs[ind]*spl.spherical_jn(l, fle.lmds[ind] * rs)*spl.sph_harm(-m, l, ps, ts)*( rs <= 1 )
            tmpm = fle.cs[ind]*spl.spherical_jn(l, fle.lmds[ind] * rs)*spl.sph_harm(m, l, ps, ts)*( rs <= 1 )
            psi2 = (tmpm - (-1)**np.abs(m)*tmpp) *fle.h*1j/2

        if m != 0:
            psi2 *= np.sqrt(2)
        psi2 = psi2.flatten()

        print(np.linalg.norm(psi0-psi2), np.linalg.norm(psi-psi2), l, k, m)

    return


def test6_visualize_eigenfunctions_for_odd_even_N():
    N = 8
    bandlimit = N
    eps = 1e-6
    fle = FLEBasis3D(N, bandlimit, eps, mode="real")
    B = fle.create_denseB()
    psi = B[:,3].reshape(N,N,N)
    plt.figure()
    plt.title("N=8 3rd eigenfuction")
    plt.imshow(np.sum(psi,axis=2))

    N = 7
    bandlimit = N
    eps = 1e-6
    fle = FLEBasis3D(N, bandlimit, eps, mode="real")
    B = fle.create_denseB()
    psi = B[:,3].reshape(N,N,N)
    plt.figure()
    plt.title("N=7 3rd eigenfuction")
    plt.imshow(np.sum(psi,axis=2))

    plt.show()



if __name__ == "__main__":
    main()
