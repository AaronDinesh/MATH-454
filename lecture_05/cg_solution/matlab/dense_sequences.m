P = [1, 2, 4, 8, 16, 32];
N0s = [2, 4, 8];

for N0 = N0s
    
    for p = P
        A = gallery('poisson', ceil(N0 * sqrt(p)));
        mmwrite(sprintf("dense_lap2D_5pt_N0%d_p%d.mtx", N0, p), full(A));
        
    end

end