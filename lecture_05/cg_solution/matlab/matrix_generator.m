n = 200;
A = gallery('poisson', n);
mmwrite(sprintf("lap2D_5pt_n%d.mtx", n), A);