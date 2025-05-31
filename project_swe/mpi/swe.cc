#include "swe.hh"
#include "xdmf_writer.hh"

#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>

#include <mpi.h>
#include "config.hh"

namespace
{

void
read_2d_array_from_DF5(const std::string &filename,
                       const std::string &dataset_name,
                       std::vector<double> &data,
                       std::size_t &nx,
                       std::size_t &ny)
{
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t dims[2];
  herr_t status;

  // Open the HDF5 file
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0)
  {
    std::cerr << "Error opening HDF5 file: " << filename << std::endl;
    return;
  }

  // Open the dataset
  dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
  if (dataset_id < 0)
  {
    std::cerr << "Error opening dataset: " << dataset_name << std::endl;
    H5Fclose(file_id);
    return;
  }

  // Get the dataspace
  dataspace_id = H5Dget_space(dataset_id);
  if (dataspace_id < 0)
  {
    std::cerr << "Error getting dataspace" << std::endl;
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }

  // Get the dimensions of the dataset
  status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  if (status < 0)
  {
    std::cerr << "Error getting dimensions" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return;
  }
  nx = dims[0];
  ny = dims[1];

  // Resize the data vector
  data.resize(nx * ny);

  // Read the data
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  if (status < 0)
  {
    std::cerr << "Error reading data" << std::endl;
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    data.clear();
    return;
  }

  // Close resources
  H5Sclose(dataspace_id);
  H5Dclose(dataset_id);
  H5Fclose(file_id);

  // std::cout << "Successfully read 2D array from HDF5 file: " << filename << ", dataset: " << dataset_name <<
  // std::endl;
}

} // namespace

SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny, MPI_Comm comm) :
  nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0)
{
  assert(test_case_id == 1 || test_case_id == 2);
  


  //Build our Cartesian Communicator
  MPI_Comm_size(comm, &this->size);
  MPI_Comm_rank(comm, &this->rank);
  dims[0] = dims[1] = 0;
  MPI_Dims_create(this->size, 2, dims);
  int periods[2] = {0,0};
  MPI_Cart_create(comm, 2, dims, periods, 1, &this->cart_comm);
  MPI_Cart_coords(this->cart_comm, this->rank, 2, this->coords);

  // This is computing my neighbors
  MPI_Cart_shift(this->cart_comm, 0, 1, &this->neighbor_west, &this->neighbor_east);
  MPI_Cart_shift(this->cart_comm, 1, 1, &this->neighbor_south, &this->neighbor_north);
  
  // This is making sure we can evenly divide the domain
  assert(nx_ % dims[0] == 0 && ny_ % dims[1] == 0);


  // Calculating the local size of my grid as well as the offsets that I need to work with
  local_nx = nx_ / dims[0];
  local_ny = ny_ / dims[1];
  offset_x = coords[0] * local_nx;
  offset_y = coords[1] * local_ny;

  //This is a function that will resize a vector to inlcude the ghost layers around each local grid. These ghost layers
  //are needed so that the cells at the boundaries of each local grid can be updated with information from its
  //neighboring grid.
  auto add_ghost = [&](std::vector<double>& V){
    V.resize((local_nx+2)*(local_ny+2), 0.0);
  };

  // Here we add all the ghost layers
  add_ghost(h0_); 
  add_ghost(h1_);
  add_ghost(hu0_); 
  add_ghost(hu1_);
  add_ghost(hv0_); 
  add_ghost(hv1_);
  add_ghost(z_);  
  add_ghost(zdx_); 
  add_ghost(zdy_);
  
  
  if (test_case_id == 1){
    this->reflective_ = true;
    this->local_init_gaussian();
  }
  else if (test_case_id == 2){
    this->reflective_ = false;
    this->local_init_dummy_tsunami();
  }
  else{
    assert(false);
  }
}

SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y) :
  size_x_(size_x), size_y_(size_y), reflective_(false)
{
  this->init_from_HDF5_file(h5_file);
}

void SWESolver::init_from_HDF5_file(const std::string &h5_file){
  read_2d_array_from_DF5(h5_file, "h0", this->h0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", this->hu0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hv0", this->hv0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "topography", this->z_, this->nx_, this->ny_);

  this->h1_.resize(this->h0_.size(), 0.0);
  this->hu1_.resize(this->hu0_.size(), 0.0);
  this->hv1_.resize(this->hv0_.size(), 0.0);

  // this->init_dx_dy();
}

// void SWESolver::init_gaussian(){
//   hu0_.resize(nx_ * ny_, 0.0);
//   hv0_.resize(nx_ * ny_, 0.0);
//   std::fill(hu0_.begin(), hu0_.end(), 0.0);
//   std::fill(hv0_.begin(), hv0_.end(), 0.0);

//   h0_.clear();
//   h0_.reserve(nx_ * ny_);

//   h1_.resize(nx_ * ny_);
//   hu1_.resize(nx_ * ny_);
//   hv1_.resize(nx_ * ny_);

//   const double x0_0 = size_x_ / 4.0;
//   const double y0_0 = size_y_ / 3.0;
//   const double x0_1 = size_x_ / 2.0;
//   const double y0_1 = 0.75 * size_y_;

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;

//   for (std::size_t j = 0; j < ny_; ++j){
//     for (std::size_t i = 0; i < nx_; ++i){
//       const double x = dx * (static_cast<double>(i) + 0.5);
//       const double y = dy * (static_cast<double>(j) + 0.5);
//       const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
//       const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

//       h0_.push_back(10.0 + gauss_0 + gauss_1);
//     }
//   }

//   z_.resize(this->h0_.size());
//   std::fill(z_.begin(), z_.end(), 0.0);

//   this->init_dx_dy();
// }

void SWESolver::exchange_halos(){
  MPI_Datatype column_dtype;
  // This vector will have local_ny elements, of block length 1, the stride between each block is local_nx+2. The base
  // unit of the vector is double.
  MPI_Type_vector(local_ny, 1, local_nx+2, MPI_DOUBLE, &column_dtype);
  MPI_Type_commit(&column_dtype);

  MPI_Datatype row_dtype;
  // This vector will have local_nx elements, of block length 1, the stride between each block is 1 (since they are
  // contiguous in memory). The base unit of the vector is double.
  MPI_Type_vector(local_nx, 1, 1, MPI_DOUBLE, &row_dtype);
  MPI_Type_commit(&row_dtype);

  // Here we create an anonymous fuctions that we perform the exchanging on all the variables.
  auto __exchange_halos = [&](std::vector<double> &var){
      MPI_Sendrecv(&at(var, 1, 1), 1, column_dtype, neighbor_west, 0,
                   &at(var, local_nx+1, 1), 1, column_dtype, neighbor_east, 0,
                   cart_comm, MPI_STATUS_IGNORE);

      MPI_Sendrecv(&at(var, local_nx, 1), 1, column_dtype, neighbor_east, 1,
                   &at(var, 0, 1), 1, column_dtype, neighbor_west, 1,
                   cart_comm, MPI_STATUS_IGNORE);

      MPI_Sendrecv(&at(var, 1, 1), 1, row_dtype, neighbor_south, 2,
                   &at(var, 1, local_ny+1), 1, row_dtype, neighbor_north, 2,
                   cart_comm, MPI_STATUS_IGNORE);  

      MPI_Sendrecv(&at(var, 1, local_ny),  1, row_dtype, neighbor_north, 3, // send your top real row
                   &at(var, 1, 0),          1, row_dtype, neighbor_south, 3, // recv into your bottom ghost
                   cart_comm, MPI_STATUS_IGNORE);
  };

  // We create a list of pointers. Then we dereference them as we pass them into __exchange_halos. __exchange_halos then
  // takes the reference of that dereferenced pointer. In this way the __exchnage_halos function is able to access and
  // modify the original vector.
  for(auto *var : {&h0_, &hu0_, &hv0_, &zdx_, &zdy_}){
    __exchange_halos(*var);
  }

  MPI_Type_free(&column_dtype);
  MPI_Type_free(&row_dtype);
}

void SWESolver::local_init_dx_dy(){
  double dx = size_x_/nx_;
  double dy = size_y_/ny_;

  // Only loop interior cells — ghosts are at 0 and local_n+1
  for(int j = 1; j <= (int) local_ny; ++j) {
    for(int i = 1; i <= (int) local_nx; ++i) {
      // central-difference in x
      at(zdx_, i, j) = ( at(z_, i+1, j) - at(z_, i-1, j) ) / (2.0*dx);

      // central-difference in y
      at(zdy_, i, j) = ( at(z_, i, j+1) - at(z_, i, j-1) ) / (2.0*dy);
    }
  }
}


void SWESolver::local_init_gaussian(){
  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  for (std::size_t j = 0; j < local_ny + 2; ++j){
    for (std::size_t i = 0; i < local_nx + 2; ++i){
      int gi = offset_x + i - 1;
      int gj = offset_y + j - 1;
      const double x = dx * (static_cast<double>(gi) + 0.5);
      const double y = dy * (static_cast<double>(gj) + 0.5);

      const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

      at(z_, i, j) = 0.0;
      at(h0_, i, j) = 10.0 + gauss_0 + gauss_1;
      at(hu0_, i, j) = 0.0;
      at(hv0_, i, j) = 0.0;
    }
  }

  // Here we exchange halos and then compute the inital derivatives
  this->exchange_halos();
  this->local_init_dx_dy();
}



// void SWESolver::init_dummy_tsunami(){
//   hu0_.resize(nx_ * ny_);
//   hv0_.resize(nx_ * ny_);
//   std::fill(hu0_.begin(), hu0_.end(), 0.0);
//   std::fill(hv0_.begin(), hv0_.end(), 0.0);

//   h1_.resize(nx_ * ny_);
//   hu1_.resize(nx_ * ny_);
//   hv1_.resize(nx_ * ny_);
//   std::fill(h1_.begin(), h1_.end(), 0.0);
//   std::fill(hu1_.begin(), hu1_.end(), 0.0);
//   std::fill(hv1_.begin(), hv1_.end(), 0.0);

//   const double x0_0 = 0.6 * size_x_;
//   const double y0_0 = 0.6 * size_y_;
//   const double x0_1 = 0.4 * size_x_;
//   const double y0_1 = 0.4 * size_y_;
//   const double x0_2 = 0.7 * size_x_;
//   const double y0_2 = 0.3 * size_y_;

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;

//   // Creating topography and initial water height
//   z_.resize(nx_ * ny_);
//   h0_.resize(nx_ * ny_);
//   for (std::size_t j = 0; j < ny_; ++j){
//     for (std::size_t i = 0; i < nx_; ++i){
//       const double x = dx * (static_cast<double>(i) + 0.5);
//       const double y = dy * (static_cast<double>(j) + 0.5);

//       const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
//       const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
//       const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

//       const double z = -1.0 + gauss_0 + gauss_1;
//       at(z_, i, j) = z;

//       double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
//       at(h0_, i, j) = h0;
//     }
//   }
//   this->init_dx_dy();
// }


void SWESolver::local_init_dummy_tsunami(){

  const double x0_0 = 0.6 * size_x_;
  const double y0_0 = 0.6 * size_y_;
  const double x0_1 = 0.4 * size_x_;
  const double y0_1 = 0.4 * size_y_;
  const double x0_2 = 0.7 * size_x_;
  const double y0_2 = 0.3 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  for(size_t j = 0; j < local_ny + 2; j++){
    for(size_t i = 0; i < local_nx + 2; i++){
      int gi = offset_x + i - 1;
      int gj = offset_y + j - 1;
      const double x = dx * (static_cast<double>(gi) + 0.5);
      const double y = dy * (static_cast<double>(gj) + 0.5);

      const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
      const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
      const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);
      const double z = -1.0 + gauss_0 + gauss_1;

      at(z_, i, j) = z;

      double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
      at(h0_, i, j) = h0;
      at(hu0_, i, j) = 0.0;
      at(hv0_, i, j) = 0.0;
      at(h1_, i, j) = 0.0;
      at(hu1_, i, j) = 0.0;
      at(hv1_, i, j) = 0.0;  
    }
  }

  // Here we share the halos and then compute the initial derivatives.
  this->exchange_halos();
  this->local_init_dx_dy();
}


// void SWESolver::init_dummy_slope(){
//   hu0_.resize(nx_ * ny_);
//   hv0_.resize(nx_ * ny_);
//   std::fill(hu0_.begin(), hu0_.end(), 0.0);
//   std::fill(hv0_.begin(), hv0_.end(), 0.0);

//   h1_.resize(nx_ * ny_);
//   hu1_.resize(nx_ * ny_);
//   hv1_.resize(nx_ * ny_);
//   std::fill(h1_.begin(), h1_.end(), 0.0);
//   std::fill(hu1_.begin(), hu1_.end(), 0.0);
//   std::fill(hv1_.begin(), hv1_.end(), 0.0);

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;

//   const double dz = 10.0;

//   // Creating topography and initial water height
//   z_.resize(nx_ * ny_);
//   h0_.resize(nx_ * ny_);
//   for (std::size_t j = 0; j < ny_; ++j){
//     for (std::size_t i = 0; i < nx_; ++i){
//       const double x = dx * (static_cast<double>(i) + 0.5);
//       const double y = dy * (static_cast<double>(j) + 0.5);
//       static_cast<void>(y);

//       const double z = -10.0 - 0.5 * dz + dz / size_x_ * x;
//       at(z_, i, j) = z;

//       double h0 = z < 0.0 ? -z : 0.00001;
//       at(h0_, i, j) = h0;
//     }
//   }
//   this->init_dx_dy();
// }

// void SWESolver::init_dx_dy(){
//   zdx_.resize(this->z_.size(), 0.0);
//   zdy_.resize(this->z_.size(), 0.0);

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;
//   for (std::size_t j = 1; j < ny_ - 1; ++j){
//     for (std::size_t i = 1; i < nx_ - 1; ++i){
//       at(this->zdx_, i, j) = 0.5 * (at(this->z_, i + 1, j) - at(this->z_, i - 1, j)) / dx;
//       at(this->zdy_, i, j) = 0.5 * (at(this->z_, i, j + 1) - at(this->z_, i, j - 1)) / dy;
//     }
//   }
// }

void SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix){
  std::shared_ptr<XDMFWriter> writer;
  if (output_n > 0){
    writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
    std::vector<double> mat = gather_data(this->cart_comm, this->rank, this->size, this->nx_, this->ny_, this->local_nx, this->local_ny, this->h0_);
    
    if(rank == 0){
      writer->add_h(mat, 0.0);
    }
  }

  double T = 0.0;

  // std::vector<double> &h = h1_;
  // std::vector<double> &hu = hu1_;
  // std::vector<double> &hv = hv1_;

  // std::vector<double> &h0 = h0_;
  // std::vector<double> &hu0 = hu0_;
  // std::vector<double> &hv0 = hv0_;
  #if DEBUG
    if(rank == 0){
      std::cout << "Solving SWE..." << std::endl;
    }
  #endif

  std::size_t nt = 1;
  while (T < Tend){
    const double dt = this->local_compute_timestep(T, Tend);

    const double T1 = T + dt;

    #if DEBUG
      if(rank == 0){
        printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
        std::cout << (full_log ? "\n" : "\r") << std::flush;
      }
    #endif
    this->exchange_halos();
    
    this->update_bcs();
    
    this->solve_step(dt);


    if (output_n > 0 && nt % output_n == 0 ){
      std::vector<double> mat = gather_data(this->cart_comm, this->rank, this->size, this->nx_, this->ny_, this->local_nx, this->local_ny, this->h1_);
      if (rank == 0){
        writer->add_h(mat, T1);
      }
    }
    ++nt;

    // Swap the old and new solutions
    std::swap(h1_, h0_);
    std::swap(hu1_, hu0_);
    std::swap(hv1_, hv0_);

    T = T1;
  }

  // Copying last computed values to h1_, hu1_, hv1_ (if needed)
  if (&h0_ != &h1_){
    h1_ = h0_;
    hu1_ = hu0_;
    hv1_ = hv0_;
  }

  if (output_n > 0){
    std::vector<double> mat = gather_data(this->cart_comm, this->rank, this->size, this->nx_, this->ny_, this->local_nx, this->local_ny, this->h1_);
    if (rank == 0){
      writer->add_h(mat, T);
    }
  }
  if(rank == 0){
    std::cout << "Finished solving SWE." << std::endl;
  }
}

// double SWESolver::compute_time_step(const std::vector<double> &h, const std::vector<double> &hu, const std::vector<double> &hv, const double T, const double Tend) const{
//   double max_nu_sqr = 0.0;
//   double au{0.0};
//   double av{0.0};
//   for (std::size_t j = 1; j < ny_ - 1; ++j)
//   {
//     for (std::size_t i = 1; i < nx_ - 1; ++i)
//     {
//       au = std::max(au, std::fabs(at(hu, i, j)));
//       av = std::max(av, std::fabs(at(hv, i, j)));
//       const double nu_u = std::fabs(at(hu, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
//       const double nu_v = std::fabs(at(hv, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
//       max_nu_sqr = std::max(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
//     }
//   }

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;
//   double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
//   return std::min(dt, Tend - T);
// }


double SWESolver::local_compute_timestep(const double T, const double Tend){
  double local_max_nu_sqr = 0.0;
  double max_nu_sqr = 0.0;
  
  for (std::size_t j = 1; j < local_ny + 1; ++j){
    for (std::size_t i = 1; i < local_nx + 1; ++i){
      const double nu_u = std::fabs(at(hu0_, i, j)) / at(h0_, i, j) + sqrt(g * at(h0_, i, j));
      const double nu_v = std::fabs(at(hv0_, i, j)) / at(h0_, i, j) + sqrt(g * at(h0_, i, j));
      local_max_nu_sqr = std::max(local_max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
    }
  }

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;


  MPI_Allreduce(&local_max_nu_sqr, &max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

  double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
  return std::min(dt, Tend - T);
}

void SWESolver::compute_kernel(const std::size_t i, const std::size_t j, const double dt){
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  const double C1x = 0.5 * dt / dx;
  const double C1y = 0.5 * dt / dy;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g;

  double hij = 0.25 * (at(h0_, i, j - 1) + at(h0_, i, j + 1) + at(h0_, i - 1, j) + at(h0_, i + 1, j))
               + C1x * (at(hu0_, i - 1, j) - at(hu0_, i + 1, j)) + C1y * (at(hv0_, i, j - 1) - at(hv0_, i, j + 1));
  
  if (hij < 0.0){
    hij = 1.0e-5;
  }

  at(h1_, i, j) = hij;

  if (hij > 0.0001){
    at(hu1_, i, j) =
      0.25 * (at(hu0_, i, j - 1) + at(hu0_, i, j + 1) + at(hu0_, i - 1, j) + at(hu0_, i + 1, j)) - C2 * hij * at(zdx_, i, j)
      + C1x
          * (at(hu0_, i - 1, j) * at(hu0_, i - 1, j) / at(h0_, i - 1, j) + C3 * at(h0_, i - 1, j) * at(h0_, i - 1, j)
             - at(hu0_, i + 1, j) * at(hu0_, i + 1, j) / at(h0_, i + 1, j) - C3 * at(h0_, i + 1, j) * at(h0_, i + 1, j))
      + C1y
          * (at(hu0_, i, j - 1) * at(hv0_, i, j - 1) / at(h0_, i, j - 1)
             - at(hu0_, i, j + 1) * at(hv0_, i, j + 1) / at(h0_, i, j + 1));

    at(hv1_, i, j) =
      0.25 * (at(hv0_, i, j - 1) + at(hv0_, i, j + 1) + at(hv0_, i - 1, j) + at(hv0_, i + 1, j)) - C2 * hij * at(zdy_, i, j)
      + C1x
          * (at(hu0_, i - 1, j) * at(hv0_, i - 1, j) / at(h0_, i - 1, j)
             - at(hu0_, i + 1, j) * at(hv0_, i + 1, j) / at(h0_, i + 1, j))
      + C1y
          * (at(hv0_, i, j - 1) * at(hv0_, i, j - 1) / at(h0_, i, j - 1) + C3 * at(h0_, i, j - 1) * at(h0_, i, j - 1)
             - at(hv0_, i, j + 1) * at(hv0_, i, j + 1) / at(h0_, i, j + 1) - C3 * at(h0_, i, j + 1) * at(h0_, i, j + 1));
  }
  else{
    at(hu1_, i, j) = 0.0;
    at(hv1_, i, j) = 0.0;
  }

  // h(2:nx-1,2:nx-1) = 0.25*(h0(2:nx-1,1:nx-2)+h0(2:nx-1,3:nx)+h0(1:nx-2,2:nx-1)+h0(3:nx,2:nx-1)) ...
  //     + C1*( hu0(2:nx-1,1:nx-2) - hu0(2:nx-1,3:nx) + hv0(1:nx-2,2:nx-1) - hvhv0:nx,2:nx-1) );

  // hu(2:nx-1,2:nx-1) = 0.25*(hu0(2:nx-1,1:nx-2)+hu0(2:nx-1,3:nx)+hu0(1:nx-2,2:nx-1)+hu0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdx(2:nx-1,2:nx-1) ...
  //     + C1*( hu0(2:nx-1,1:nx-2).^2./h0(2:nx-1,1:nx-2) + 0.5*g*h0(2:nx-1,1:nx-2).^2 -
  //     hu0(2:nx-1,3:nx).^2./h0(2:nx-1,3:nx) - 0.5*g*h0(2:nx-1,3:nx).^2 ) ...
  //     + C1*( hu0(1:nx-2,2:nx-1).*hv0(1:nx-2,2:nx-1)./h0(1:nx-2,2:nx-1) -
  //     hu0(3:nx,2:nx-1).*hv0(3:nx,2:nx-1)./h0(3:nx,2:nx-1) );

  // hv(2:nx-1,2:nx-1) = 0.25*(hv0(2:nx-1,1:nx-2)+hv0(2:nx-1,3:nx)+hv0(1:nx-2,2:nx-1)+hv0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdy(2:nx-1,2:nx-1)  ...
  //     + C1*( hu0(2:nx-1,1:nx-2).*hv0(2:nx-1,1:nx-2)./h0(2:nx-1,1:nx-2) -
  //     hu0(2:nx-1,3:nx).*hv0(2:nx-1,3:nx)./h0(2:nx-1,3:nx) ) ...
  //     + C1*( hv0(1:nx-2,2:nx-1).^2./h0(1:nx-2,2:nx-1) + 0.5*g*h0(1:nx-2,2:nx-1).^2 -
  //     hv0(3:nx,2:nx-1).^2./h0(3:nx,2:nx-1) - 0.5*g*h0(3:nx,2:nx-1).^2  );
}

void SWESolver::solve_step(const double dt){
  for (std::size_t j = 1; j < local_ny + 1; ++j){  
    for (std::size_t i = 1; i < local_nx + 1; ++i){
      this->compute_kernel(i, j, dt);
    }
  }
}


std::vector<double> SWESolver::gather_data(const MPI_Comm cart_comm, int rank, int size, std::size_t nx_, std::size_t ny_, std::size_t local_nx, std::size_t local_ny, const std::vector<double> &h0_){
  std::vector<double> send_buffer(local_nx*local_ny);

  for (std::size_t j = 0; j < local_ny; ++j){
    for (std::size_t i = 0; i < local_nx; ++i){
      send_buffer[j * local_nx + i] = h0_[((j + 1) * (local_nx + 2) + (i + 1))];
    }
  }


  // Every proc needs to have this recv buffer variable
  std::vector<double> recv_buff;

  if (rank == 0){
    // Only proc 0 needs to resize the recv buffer
    recv_buff.resize(static_cast<std::size_t>(size) * local_nx * local_ny);
  }

  //Now we do an MPI_Gather to 0.
  MPI_Gather(send_buffer.data(), 
            static_cast<int>(local_nx * local_ny), // Each rank needs to send this many elements
            MPI_DOUBLE,
            recv_buff.data(),
            static_cast<int>(local_nx * local_ny), // Proc 0 will receive this many elements from each of the other  ranks
            MPI_DOUBLE, 0, cart_comm);

  std::vector<double> global_matrix;
  if(rank == 0){
    std::vector<double> global_matrix(nx_ * ny_, 0.0);

    for(int _rank = 0; _rank < size; ++_rank){
      int coords[2];
      MPI_Cart_coords(cart_comm, _rank, 2, coords);
      
      int off_x = coords[0] * static_cast<int>(local_nx);
      int off_y = coords[1] * static_cast<int>(local_ny);

      double* block_recv_buff = recv_buff.data() + static_cast<std::size_t>(_rank) * (local_nx * local_ny);

      for(std::size_t j = 0; j < local_ny; ++j){
        for(std::size_t i = 0; i < local_nx; ++i){
          std::size_t global_i = static_cast<std::size_t>(off_x) + i;
          std::size_t global_j = static_cast<std::size_t>(off_y) + j;
          global_matrix[global_j * nx_ + global_i] = block_recv_buff[j * local_nx + i];
        }
      }
    }
  }
  MPI_Barrier(cart_comm);
  return global_matrix;
}


void SWESolver::update_bcs(){
  const double coef = this->reflective_ ? -1.0 : 1.0;

  //figure out which boundary I am. One of my neighbours will be MPI_PROC_NULL. If two of my neighbours are
  //MPI_PROC_NULL, I am a corner.

  if (neighbor_west == MPI_PROC_NULL){
    // This is the west side boundary

    for (std::size_t j = 1; j <= local_ny; ++j){
      at(h1_, 0, j) = at(h0_, 1, j);
      at(hu1_, 0, j) = coef * at(hu0_, 1, j);
      at(hv1_, 0, j) = at(hv0_, 1, j);
    }
  }

  if (neighbor_east== MPI_PROC_NULL){
    // This is the east side boundary
    for (std::size_t j = 1; j <= local_ny; ++j){
    at(h1_, local_nx+1, j) = at(h0_, local_nx, j);
    at(hu1_, local_nx+1, j) = coef * at(hu0_, local_nx, j);
    at(hv1_, local_nx+1, j) = at(hv0_, local_nx, j);
    }
  }

  if (neighbor_south == MPI_PROC_NULL){
    // This is the north side boundary
    for (std::size_t i = 1; i <= local_nx; ++i){
      at(h1_, i, 0) = at(h0_, i, 1);
      at(hu1_, i, 0) = at(hu0_, i, 1);
      at(hv1_, i, 0) = coef * at(hv0_, i, 1);
    }
  }

  if (neighbor_north == MPI_PROC_NULL){
    // This is the south side boundary
    for (std::size_t i = 1; i <= local_nx; ++i){
      at(h1_, i, local_ny + 1) = at(h0_, i, local_ny);
      at(hu1_, i, local_ny + 1) = at(hu0_, i, local_ny);
      at(hv1_, i, local_ny + 1) = coef * at(hv0_, i, local_ny);
    }
  }
};