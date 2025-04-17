P = [1, 2, 4, 8, 16, 32];
N0s = [1024, 2048, 4096];

for N0 = N0s
    
    for p = P
        A = gallery('poisson', ceil(sqrt(N0 * p)));
        mmwrite(sprintf("lap2D_5pt_N0%d_p%d.mtx", N0, p), A);
        
    end

end