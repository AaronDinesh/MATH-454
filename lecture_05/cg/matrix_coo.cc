#include "matrix_coo.hh"
extern "C" {
#include "mmio.h"
}
#include <mpi.h>
#include <cassert>
#include <functional>
#include <algorithm>
#include <stddef.h>
#include <iostream>

void MatrixCOO::read(const std::string & fn) {
  int nz;

  int ret_code;
  MM_typecode matcode;
  FILE * f;

  if ((f = fopen(fn.c_str(), "r")) == NULL) {
    printf("Could not open matrix");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Matrix is sparse
  if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    exit(1);
  }

  /* reserve memory for matrices */
  irn.resize(nz);
  jcn.resize(nz);
  a.resize(nz);

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  m_is_sym = mm_is_symmetric(matcode);
  for (int i = 0; i < nz; i++) {
    int I, J;
    double a_;

    int len =  fscanf(f, "%d %d %lg\n", &I, &J, &a_);

    if(len < 0){
      std::cerr << "Error reading matrix" << std::endl;
      MPI_Finalize();
    }

    I--; /* adjust from 1-based to 0-based */
    J--;

    irn[i] = I;
    jcn[i] = J;
    a[i] = a_;
  }

  if (f != stdin) {
    fclose(f);
  }
}


void MatrixCOO::read_distributed(const std::string& filename, MPI_Comm comm) {
  int size, rank;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int nz;
  std::vector<int> I, J;
  std::vector<double> val;
  
  //Create out counts and displacement vector in preparation for ScatterV
  std::vector<int> counts(size), displacements(size);

  if(rank == 0){
    int *I_raw, *J_raw;
    double *val_raw;
    MM_typecode matcode;

    // int ret_code;
    // FILE* f;

    // if ((f = fopen(filename.c_str(), "r")) == NULL) {
    //   printf("Could not open matrix");
    //   exit(1);
    // }

    // if (mm_read_banner(f, &matcode) != 0) {
    //   printf("Could not process Matrix Market banner.\n");
    //   exit(1);
    // }

    // // Matrix is sparse
    // if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    //   printf("Sorry, this application does not support ");
    //   printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    //   exit(1);
    // }

    // if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    //   exit(1);
    // }

    int status = mm_read_mtx_crd(const_cast<char *>(filename.c_str()), &m_m, &m_n, &nz, &I_raw, &J_raw, &val_raw, &matcode);
    
    if(nz <= size){
      fprintf(stderr, "Error: There are more MPI nodes than there are matrix values. Reduce the number of nodes and run again. The number of nodes should be strictly less than the number of matrix values\n");
      MPI_Abort(comm, MPI_ERR_SIZE);
    }

    //Checks if the matrix is symmetric
    m_is_sym = mm_is_symmetric(matcode);

    if (status != 0) {
      fprintf(stderr, "Error: Failed to read Matrix Market file '%s'\n", filename.c_str());
      MPI_Abort(comm,  MPI_ERR_NO_SUCH_FILE);
    }

    I.assign(I_raw, I_raw + nz);
    J.assign(J_raw, J_raw + nz);
    val.assign(val_raw, val_raw + nz);
    free(I_raw);
    free(J_raw);
    free(val_raw);

    // if(f != stdin){
    //   //Close the file after we have finished reading
    //   fclose(f);
    // }
    
    //Compute the rough division and then the remainder
    int base = nz / size;
    int rem = nz % size;
    
    //Compute the displacement and the counts for each processor
    for (int i = 0; i < size; ++i) {
      counts[i] = base + (i < rem ? 1 : 0);
      displacements[i] = (i == 0) ? 0 : displacements[i - 1] + counts[i - 1];
    }

  }

  //Processor 0 needs to pack all the data and send it out
  int packed_meta_size = 0;
  int packed_bool_size = 0;
  int counts_disp_size = 0;
  MPI_Pack_size(3, MPI_INT, comm, &packed_meta_size);
  MPI_Pack_size(1, MPI_CXX_BOOL, comm, &packed_bool_size);
  MPI_Pack_size(2 * size, MPI_INT, comm, &counts_disp_size);

  int meta_buf_size = packed_meta_size + packed_bool_size + counts_disp_size;
  std::vector<char> meta_buffer(meta_buf_size);
  int meta_pos = 0;

  if (rank == 0){
    // == 3 Ints ==
    MPI_Pack(&m_m, 1, MPI_INT, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
    MPI_Pack(&m_n, 1, MPI_INT, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
    MPI_Pack(&nz, 1, MPI_INT, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
    
    // == 2 arrays of size mpi-size ==
    MPI_Pack(counts.data(), size, MPI_INT, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
    MPI_Pack(displacements.data(), size, MPI_INT, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
    
    // == 1 bool ==
    MPI_Pack(&m_is_sym, 1, MPI_CXX_BOOL, meta_buffer.data(), meta_buf_size, &meta_pos, comm);
  }

  //Broadcast the data
  MPI_Bcast(meta_buffer.data(), meta_pos, MPI_PACKED, 0, comm);

  // Unpack all the data
  int unpack_pos = 0;
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, &m_m, 1, MPI_INT, comm);
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, &m_n, 1, MPI_INT, comm);
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, &nz , 1, MPI_INT, comm);
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, counts.data(), size, MPI_INT, comm);
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, displacements.data(), size, MPI_INT, comm);
  MPI_Unpack(meta_buffer.data(), meta_pos, &unpack_pos, &m_is_sym, 1, MPI_CXX_BOOL, comm);

  irn.resize(counts[rank]);
  jcn.resize(counts[rank]);
  a.resize(counts[rank]);

  int int_size = 0;
  int double_size = 0;
  MPI_Pack_size(1, MPI_INT, comm, &int_size);
  MPI_Pack_size(1, MPI_DOUBLE, comm, &double_size);
  int per_entry_size = 2 * int_size + double_size;

  //send_counts is the number of bytes we send to each processor
  std::vector<int> send_counts(size), send_displs(size);
  //This is the raw array of bytes
  std::vector<char> send_buffer;

  if (rank == 0) {
    for (int i = 0; i < size; ++i)
      send_counts[i] = counts[i] * per_entry_size;

    send_displs[0] = 0;
    for (int i = 1; i < size; ++i)
      send_displs[i] = send_displs[i - 1] + send_counts[i - 1];

    send_buffer.resize(send_displs.back() + send_counts.back());

    for (int i = 0; i < size; ++i) {
      int pos = 0;
      char* ptr = send_buffer.data() + send_displs[i];
      int offset = displacements[i];
      for (int j = 0; j < counts[i]; ++j) {
        int idx = offset + j;
        MPI_Pack(&I[idx], 1, MPI_INT, ptr, send_counts[i], &pos, comm);
        MPI_Pack(&J[idx], 1, MPI_INT, ptr, send_counts[i], &pos, comm);
        MPI_Pack(&val[idx], 1, MPI_DOUBLE, ptr, send_counts[i], &pos, comm);
      }
    }
  }

  std::vector<char> recv_buffer(counts[rank] * per_entry_size);
  MPI_Scatterv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_PACKED,
               recv_buffer.data(), recv_buffer.size(), MPI_PACKED, 0, comm);

  // Unpack into local COO storage
  int local_nz = counts[rank];
  irn.resize(local_nz);
  jcn.resize(local_nz);
  a.resize(local_nz);

  int pos = 0;
  for (int i = 0; i < local_nz; ++i) {
    MPI_Unpack(recv_buffer.data(), recv_buffer.size(), &pos, &irn[i], 1, MPI_INT, comm);
    MPI_Unpack(recv_buffer.data(), recv_buffer.size(), &pos, &jcn[i], 1, MPI_INT, comm);
    MPI_Unpack(recv_buffer.data(), recv_buffer.size(), &pos, &a[i], 1, MPI_DOUBLE, comm);
  }
}


void MatrixCOO::mat_vec(const std::vector<double>& x, std::vector<double>& result) {
  std::fill_n(result.begin(), result.size(), 0.);

  for (size_t z = 0; z < irn.size(); ++z) {
    int i = irn[z];
    int j = jcn[z];
    double a_ = a[z];

    result[i] += a_ * x[j];

    if (m_is_sym and (i != j)) {
      result[j] += a_ * x[i];
    }
  }
}


MatrixCOO MatrixCOO::operator-(const MatrixCOO& rhs) { 
  assert(m_m == rhs.m_m && m_n == rhs.m_n && "Cannot add matrices with different dimensions");
  


  //Create the return struct
  MatrixCOO res;
  res.m_m = m_m;
  res.m_n = m_n;
  //The resulting matrix is only symmetric if both matrices are
  res.m_is_sym = m_is_sym & rhs.m_is_sym;

  int i =0;
  int j = 0;

  // Go over the main body of the coordinate arrays
  while(i < static_cast<int>(this->a.size()) && j < static_cast<int>(rhs.a.size())){
    if(this->irn[i] == rhs.irn[j] && this->jcn[i] == rhs.jcn[j]){
      res.irn.push_back(this->irn[i]);
      res.jcn.push_back(this->jcn[i]);
      res.a.push_back(this->a[i] - rhs.a[j]);
      i++;
      j++;
    } else if(this->irn[i] < rhs.irn[j] || (this->irn[i] == rhs.irn[j] && this->jcn[i] < rhs.jcn[j])){
      res.irn.push_back(this->irn[i]);
      res.jcn.push_back(this->jcn[i]);
      res.a.push_back(this->a[i]);
      i++;
    } else {
      res.irn.push_back(rhs.irn[j]);
      res.jcn.push_back(rhs.jcn[j]);
      res.a.push_back(-rhs.a[j]);
      j++;
    }
  }

  //The next two while loops are just to handle the remaining elements from both matrices if there are any
  while(i < static_cast<int>(this->a.size())){
    res.irn.push_back(this->irn[i]);
    res.jcn.push_back(this->jcn[i]);
    res.a.push_back(this->a[i]);
    i++;
  }

  while(j < static_cast<int>(rhs.a.size())){
    res.irn.push_back(rhs.irn[j]);
    res.jcn.push_back(rhs.jcn[j]);
    res.a.push_back(-rhs.a[j]);
    j++;
  }

  return res;
}


MatrixCOO MatrixCOO::operator+(const MatrixCOO& rhs) {
  assert(m_m == rhs.m_m && m_n == rhs.m_n && "Cannot add matrices with different dimensions");
  
  
  //Create the return struct
  MatrixCOO res;
  res.m_m = m_m;
  res.m_n = m_n;
  res.m_is_sym = m_is_sym & rhs.m_is_sym;

  int i = 0;
  int j = 0;

  // Go over the main body of the coordinate arrays
  while(i < static_cast<int>(this->a.size()) && j < static_cast<int>(rhs.a.size())){
    if(this->irn[i] == rhs.irn[j] && this->jcn[i] == rhs.jcn[j]){
      res.irn.push_back(this->irn[i]);
      res.jcn.push_back(this->jcn[i]);
      res.a.push_back(this->a[i] + rhs.a[j]);
      i++;
      j++;
    } else if(this->irn[i] < rhs.irn[j] || (this->irn[i] == rhs.irn[j] && this->jcn[i] < rhs.jcn[j])){
      res.irn.push_back(this->irn[i]);
      res.jcn.push_back(this->jcn[i]);
      res.a.push_back(this->a[i]);
      i++;
    } else {
      res.irn.push_back(rhs.irn[j]);
      res.jcn.push_back(rhs.jcn[j]);
      res.a.push_back(rhs.a[j]);
      j++;
    }
  }

  //The next two while loops are just to handle the remaining elements from both matrices if there are any
  while(i < static_cast<int>(this->a.size())){
    res.irn.push_back(this->irn[i]);
    res.jcn.push_back(this->jcn[i]);
    res.a.push_back(this->a[i]);
    i++;
  }

  while(j < static_cast<int>(rhs.a.size())){
    res.irn.push_back(rhs.irn[j]);
    res.jcn.push_back(rhs.jcn[j]);
    res.a.push_back(rhs.a[j]);
    j++;
  }

  return res;
}