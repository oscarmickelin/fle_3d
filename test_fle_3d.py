import time
from os.path import exists
from scipy.io import savemat
from fle_3d import FLEBasis3D
import numpy as np
from scipy.io import loadmat
import mrcfile



def main():
    #######
    # If True, reduces the number of radial points in defining
    # NUFFT grids, and does an alternative interpolation to
    # compensate. To reproduce the tables and figures of the
    # paper, set this to False. 
    reduce_memory = True
    #######

    # test 1: Verify that code agrees with dense matrix mulitplication
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
    # # print('... testing FastTransforms.jl and real mode')
    # # test1_fle_vs_dense("FastTransforms.jl", "real")
    # # print('... testing FastTransforms.jl and complex mode')
    # # test1_fle_vs_dense("FastTransforms.jl", "complex")
    # ##########################

    # # test 2: verify that code can lowpass
    print("test 2")
    test2_fle_lowpass(reduce_memory)

    # # # test 3: verify timing 
    print("test 3")
    test3_part_timing(reduce_memory)

    # # # test 4: check the error of
    # # least-squares expansions into the basis
    print("test 4")
    test4_expand_error_test(reduce_memory)

    return

def test1_fle_vs_dense(sph_harm_solver,mode, reduce_memory):

    Ns = [32]
    ls = []
    epss = []
    
    for l in Ns:        
    	for eps in (1e-4, 1e-7, 1e-10, 1e-14):
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    erra = np.zeros(n)
    errx = np.zeros(n)
    erra2 = np.zeros(n)
    errx2 = np.zeros(n)
    
    i = 0
    for l in Ns:
        print('Precomputing FLE...')
        bandlimit = l
        fle = FLEBasis3D(l, bandlimit, 1e-4, sph_harm_solver=sph_harm_solver,mode=mode,reduce_memory=reduce_memory)
        print('Creating dense matrix...')
        B = fle.create_denseB(numthread=1)
        print('... dense matrix created')
        for eps in (1e-4, 1e-7, 1e-10, 1e-14):
            tmperra, tmperrx, tmperra2, tmperrx2 = test1_fle_vs_dense_helper(sph_harm_solver,mode,l, eps, B, reduce_memory)
            erra[i] = tmperra
            errx[i] = tmperrx
            erra2[i] = tmperra2
            errx2[i] = tmperrx2
            i += 1


    # make {tab:accuracy}
    print(sph_harm_solver,mode)
    print()
    print(r"\begin{tabular}{r|ccc}")
    print("$N$ & $\epsilon$ & $\\text{err}_a$ & $\\text{err}_f$& $\\text{l2 err}_a$ & $\\text{l2 err}_f$ \\\\")
    print(r"\hline")
    for i in range(n):
        print(
            ls[i],
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


def test3_part_timing():

    nr = 1  # number of trials
    Ns = [32,48,64,128,256]
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
    f = f.flatten()

    # Step 1. {sec:fast_details}
    t0 = time.time()
    z = fle.step1(f)
    t1 = time.time()
    dt1 = t1 - t0

    # Step 2. {sec:fast_details}
    t0 = time.time()
    b = fle.step2(z)
    t1 = time.time()
    dt2 = t1 - t0

    # Step 3: {sec:fast_details}
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

    ls = []
    epss = []
    for eps in (1e-4, 1e-7, 1e-10, 1e-14):
        for l in [32,48,56,64,128,256]:
            ls.append(l)
            epss.append(eps)
    n = len(ls)
    err = np.zeros(n)
    err2 = np.zeros(n)
    for i in range(n):
        err[i], err2[i] = test4_helper(ls[i], epss[i], reduce_memory)

    # make {tab:accuracy}
    print("expand test")
    for i in range(n):
        print(
            ls[i],
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
    bandlimit = N
    fle = FLEBasis3D(N, bandlimit, eps, mode="complex", reduce_memory=reduce_memory)

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







if __name__ == "__main__":
    main()
