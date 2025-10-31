#include "mips/index_mips.h"

#include <omp.h>
#include <bitset>
#include <chrono>
#include <cmath>
#include <queue>
#include <boost/dynamic_bitset.hpp>

#include "ssg/parameters.h"

constexpr double kPi = 3.14159265358979323846264;

namespace efanna2e {
#define _CONTROL_NUM 100

IndexMips::IndexMips(const size_t dimension, const size_t n, Metric m,
                     Index *initializer)
    : Index(dimension, n, m), initializer_(initializer) {
    }

IndexMips::~IndexMips() {}

void IndexMips::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);
  out.write((char *)&width, sizeof(unsigned));
  unsigned n_ep=eps_.size();
  out.write((char *)&n_ep, sizeof(unsigned));
  out.write((char *)eps_.data(), n_ep*sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned)final_graph_[i].size();
    // std::cout << GK << std::endl;
    out.write((char *)&GK, sizeof(unsigned));
    out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexMips::Load(const char *filename) {
  /* width | eps_(vector<unsigned) entries | k (neighbor size) | k * unsigned (neighbors)*/
  std::ifstream in(filename, std::ios::binary);
  in.read((char *)&width, sizeof(unsigned));
  unsigned n_ep=0;
  in.read((char *)&n_ep, sizeof(unsigned));
  eps_.resize(n_ep);
  in.read((char *)eps_.data(), n_ep*sizeof(unsigned));
  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  DistanceFastL2 *norm_dis = (DistanceFastL2 *)distance_;
  norms_.resize(nd_);
  for (size_t i = 0; i < nd_; i++) {
    norms_[i] = std::sqrt(norm_dis->norm(data_ + dimension_ * i, dimension_)) ;
  }

  std::cerr << "Average Degree = " << cc << std::endl;
}

void IndexMips::Load_nn_graph(const char *filename) {
  /* k (k neighbor)| num * ( id + k * id)*/
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  std::cout << std::string(filename) << std::endl;
  in.read((char *)&k, sizeof(unsigned));
  std::cout << k << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  size_t num = (unsigned)(fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);
  std::cout << num << std::endl;
  final_graph_.resize(num);
  final_graph_.reserve(num);
  unsigned kk = (k + 3) / 4 * 4;
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    final_graph_[i].reserve(kk);
    in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

/* select l neighbors candidate from 2-hop*/
void IndexMips::get_neighbors(const unsigned q, const Parameters &parameter,
                             std::vector<Neighbor> &pool, boost::dynamic_bitset<> &flags) {
  // boost::dynamic_bitset<> flags{nd_, 0};
  unsigned L = parameter.Get<unsigned>("L");
  flags[q] = true;
  for (unsigned i = 0; i < final_graph_[q].size(); i++) {
    unsigned nid = final_graph_[q][i];
    for (unsigned nn = 0; nn < final_graph_[nid].size(); nn++) {
      unsigned nnid = final_graph_[nid][nn];
      if (flags[nnid]) continue;
      flags[nnid] = true;
      // float dist = 0;
      float dist = distance_->compare(data_ + dimension_ * q,
                                      data_ + dimension_ * nnid, dimension_);
      pool.push_back(Neighbor(nnid, dist, true));
      if (pool.size() >= L) break;
    }
    if (pool.size() >= L) break;
  }
}

void IndexMips::get_neighbors(const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = 1;
    L++;
  }
  /* sort the inital return set*/
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) { /* until no new node*/
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k) /* back to new node's neighbors*/
      k = nk;
    else
      ++k;
  }
}

void IndexMips::get_refine_neighbors(const unsigned ep, const float *query, const Parameters &parameter,
                             std::vector<Neighbor> &retset,
                             std::vector<Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("M");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  boost::dynamic_bitset<> visit_flags{nd_, 0};

  L = 0;

  for (unsigned i = 0; i < final_graph_[ep].size(); i++) {
    visit_flags[final_graph_[ep][i]] = true;
    float dist = distance_->compare(data_ + dimension_ * (size_t)final_graph_[ep][i], query,
                                    (unsigned)dimension_);
    fullset.push_back(Neighbor(final_graph_[ep][i], dist, true));
  }

  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep].size(); i++) {
    init_ids[i] = final_graph_[ep][i];
    flags[init_ids[i]] = true;
    L++;
  }

  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = Neighbor(id, dist, true);
  }
  /* sort the inital return set*/
  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        if (dist >= retset[L - 1].distance) continue;

        if (!visit_flags[id]) {
            fullset.push_back(nn);
        }
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}

void IndexMips::get_mips_neighbors(const unsigned ep, const float *query, const Parameters &parameter,
                             std::vector<IpNeighbor> &retset) {
  unsigned L = parameter.Get<unsigned>("M");
  float threshold = std::cos(40 / 180 * kPi);
  retset.resize(L + 1);
  std::vector<IpNeighbor> ipset;
  ipset.resize(L + 1);
  
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};

  L = 0;

  DistanceInnerProduct *dis_inner = new DistanceInnerProduct();

  for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep].size(); i++) {
    init_ids[i] = final_graph_[ep][i];
    flags[init_ids[i]] = true;
    L++;
  }

  while (L < init_ids.size()) {
    unsigned id = rand() % nd_;
    if (flags[id]) continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }
  int count = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if (id >= nd_) continue;
    // std::cout<<id<<std::endl;
    float dist = dis_inner->compare(data_ + dimension_ * (size_t)id, query,
                                    (unsigned)dimension_);
    retset[i] = IpNeighbor(id, dist, true);
    auto cos = dist / norms_[id] / norms_[ep];
    if (cos >= threshold) {
      ipset[count] = IpNeighbor(id, dist, true);
      count++;
    }

  }
  /* sort the inital return set*/
  std::sort(retset.begin(), retset.begin() + L);
  std::sort(ipset.begin(), ipset.begin() + count);
  int k = 0;
  while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;
    
        float dist = dis_inner->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        
        IpNeighbor nn(id, dist, true);
        if (dist <= retset[L - 1].distance) continue;

        int r = InsertIntoIpPool(retset.data(), L, nn);

        auto cos = dist / norms_[id] / norms_[ep];
        if (cos >= threshold) {
          int ir = InsertIntoIpPool(ipset.data(), count, nn);
          if (count + 1 < ipset.size()) ++count;
        }

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
}



/* find the entry node */
void IndexMips::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector<Neighbor> tmp, pool;
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;  
  std::cout << ep_ << std::endl;
  DistanceFastL2 *norm_dis = (DistanceFastL2 *)distance_;
  norms_.resize(nd_);
  for (size_t i = 0; i < nd_; i++) {
    norms_[i] = std::sqrt(norm_dis->norm(data_ + dimension_ * i, dimension_)) ;
  }
}


void IndexMips::sync_prune(unsigned q, std::vector<Neighbor> &pool,
                          const Parameters &parameters, float threshold,
                          SimpleNeighbor *cut_graph_) {
  unsigned range = parameters.Get<unsigned>("R"); /* max neighbor size (after pruning)*/
  width = range;
  unsigned start = 0;

  boost::dynamic_bitset<> flags{nd_, 0};
  for (unsigned i = 0; i < pool.size(); ++i) {
    flags[pool[i].id] = 1;
  }

  // std::cout << pool.size() << std::endl;
  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id]) continue;
    float dist = distance_->compare(data_ + dimension_ * (size_t)q,
                                    data_ + dimension_ * (size_t)id,
                                    (unsigned)dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector<Neighbor> result;
  if (pool[start].id == q) start++;
  result.push_back(pool[start]);

  auto norm = norms_[q];
  while (result.size() < range && (++start) < pool.size()) {
    auto &p = pool[start];
    bool occlude = false;
    auto p_norm = norms_[p.id];
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id,
                                     data_ + dimension_ * (size_t)p.id,
                                     (unsigned)dimension_);
      float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                     sqrt(p.distance * result[t].distance);
      if (cos_ij > threshold) {
        occlude = true;
        break;
      }
    }
    if (!occlude) result.push_back(p);
  }

  SimpleNeighbor *des_pool = cut_graph_ + (size_t)q * (size_t)range;
  for (size_t t = 0; t < result.size(); t++) {
    des_pool[t].id = result[t].id;
    des_pool[t].distance = result[t].distance;
  }
  if (result.size() < range) {
    des_pool[result.size()].distance = -1;
  }
}

void IndexMips::InterInsert(unsigned n, unsigned range, float threshold,
                           std::vector<std::mutex> &locks,
                           SimpleNeighbor *cut_graph_) {
  SimpleNeighbor *src_pool = cut_graph_ + (size_t)n * (size_t)range;
  for (size_t i = 0; i < range; i++) {
    if (src_pool[i].distance == -1) break;

    SimpleNeighbor sn(n, src_pool[i].distance);
    size_t des = src_pool[i].id;
    SimpleNeighbor *des_pool = cut_graph_ + des * (size_t)range;

    std::vector<SimpleNeighbor> temp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (size_t j = 0; j < range; j++) {
        if (des_pool[j].distance == -1) break;
        if (n == des_pool[j].id) {
          dup = 1;
          break;
        }
        temp_pool.push_back(des_pool[j]);
      }
    }
    if (dup) continue;

    temp_pool.push_back(sn);

    if (temp_pool.size() > range) {
      std::vector<SimpleNeighbor> result;
      unsigned start = 0;
      std::sort(temp_pool.begin(), temp_pool.end());
      result.push_back(temp_pool[start]);
      while (result.size() < range && (++start) < temp_pool.size()) {
        auto &p = temp_pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = distance_->compare(
              data_ + dimension_ * (size_t)result[t].id,
              data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
          float cos_ij = (p.distance + result[t].distance - djk) / 2 /
                         sqrt(p.distance * result[t].distance);
          if (cos_ij > threshold) {
            occlude = true;
            break;
          }
        }
        if (!occlude) result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (unsigned t = 0; t < result.size(); t++) {
          des_pool[t] = result[t];
        }
        if (result.size() < range) {
          des_pool[result.size()].distance = -1;
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (unsigned t = 0; t < range; t++) {
        if (des_pool[t].distance == -1) {
          des_pool[t] = sn;
          if (t + 1 < range) des_pool[t + 1].distance = -1;
          break;
        }
      }
    }
  }
}

void IndexMips::add_mips_neighbors(const unsigned n, const Parameters &parameters, std::vector<IpNeighbor> &pool, SimpleNeighbor *cut_graph_) {
  unsigned range = parameters.Get<unsigned>("R");
  SimpleNeighbor *des_pool = cut_graph_ + (size_t)n * (size_t)range;
  int count = 0;
  // std::cout << pool.size() << std::endl;
  while (des_pool[count].distance != -1 && count < range) count++;
  for (size_t t = 0; t < pool.size(); t++) {
    if (count >= range) break;
    des_pool[count].id = pool[t].id;
    des_pool[count].distance = pool[t].distance;
    count++;
  }
  if (count < range) {
    des_pool[count].distance = -1;
  }
}

void IndexMips::Link(const Parameters &parameters, SimpleNeighbor *cut_graph_) {
    unsigned range = parameters.Get<unsigned>("R");
    std::vector<std::mutex> locks(nd_);
    float angle = parameters.Get<float>("A");
    float threshold = std::cos(angle / 180 * kPi);

    omp_set_num_threads(48);
    #pragma omp parallel
    {
        std::vector<Neighbor> pool, tmp;
        boost::dynamic_bitset<> flags{nd_, 0};

        #pragma omp for schedule(dynamic, 64)  
        for (unsigned n = 0; n < nd_; ++n) {
            flags.reset();
            pool.clear();
            tmp.clear();
            get_neighbors(n, parameters, pool, flags);
            sync_prune(n, pool, parameters, threshold, cut_graph_);
        }
    }
    std::cout << "sync prune done!" << std::endl;

    omp_set_num_threads(48);
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 64)  
        for (unsigned n = 0; n < nd_; ++n) {
            InterInsert(n, range, threshold, locks, cut_graph_);
        }
    }
    std::cout << "interinsert done!" << std::endl;

    omp_set_num_threads(48);
    #pragma omp parallel
    {
        std::vector<IpNeighbor> ip_pool;

        #pragma omp for schedule(dynamic, 64) 
        for (unsigned n = 0; n < nd_; ++n) {
            ip_pool.clear();
            get_mips_neighbors(n, data_ + dimension_ * n, parameters, ip_pool);
            add_mips_neighbors(n, parameters, ip_pool, cut_graph_);
        }
    }
    std::cout << "add mips neighbors done!" << std::endl;
}

void IndexMips::Build(size_t n, const float *data,
                     const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  unsigned range = parameters.Get<unsigned>("R");
  Load_nn_graph(nn_graph_path.c_str());
  std::cout <<"Load nn graph" << std::endl;
  data_ = data;
  init_graph(parameters);
  std::cout << "Init done" << std::endl;
  SimpleNeighbor *cut_graph_ = new SimpleNeighbor[nd_ * (size_t)range];
  Link(parameters, cut_graph_);
  std::cout << "Link done" << std::endl;
  
  final_graph_.resize(nd_);

  for (size_t i = 0; i < nd_; i++) {
    SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
    unsigned pool_size = 0;
    for (unsigned j = 0; j < range; j++) {
      if (pool[j].distance == -1) {
        break;
      }
      pool_size = j;
    }
    ++pool_size;
    final_graph_[i].resize(pool_size);
    for (unsigned j = 0; j < pool_size; j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }

  // DFS_expand(parameters);


  unsigned max, min, avg;
  max = 0;
  min = nd_;
  avg = 0;
  for (size_t i = 0; i < nd_; i++) {
    auto size = final_graph_[i].size();
    max = max < size ? size : max;
    min = min > size ? size : min;
    avg += size;
  }
  avg /= 1.0 * nd_;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n",
         max, min, avg);

  has_built = true;
}

void IndexMips::Search(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
}


int IndexMips::Search_Mips_IP_Cal(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices, std::vector<int> &init_nodes) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  auto dis_cal = 0;
  data_ = x;
  int update_position = 0;
  std::vector<IpNeighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
  // assert(eps_.size() < L);
  for(unsigned i=0; i<eps_.size(); i++){
    init_ids[i] = eps_[i];
  }
  
  for (unsigned i = 0; i < init_nodes.size() && i < L; i++) {
    init_ids[i] = init_nodes[i];
  }
  DistanceInnerProduct dist_inner;
  for (unsigned i = 0; i < L; i++) {
    unsigned id = init_ids[i];
    float ip = dist_inner.compare(data_ + dimension_ * id, query,
                                    dimension_);
    retset[i] = IpNeighbor(id, ip, true);
    flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  
  int k = 0;
  while (k < (int)L) {
    int nk = L;
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      int search_pos = final_graph_[n].size();
      for (unsigned m = 0; m < search_pos; ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;        
        float dist = dist_inner.compare(query, data_ + dimension_ * id,
                                        dimension_); 
        dis_cal += 1;
        if (dist <= retset[L - 1].distance) continue;
        IpNeighbor nn(id, dist, true);
        int r = InsertIntoIpPool(retset.data(), L, nn);
        if (r < nk) {
          nk = r;
        }
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
  return {dis_cal};
}

int IndexMips::Search_Mips_IP_Cal_with_No_SN(const float *query, const float *x, size_t K,
                      const Parameters &parameters, unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  auto dis_cal = 0;
  data_ = x;
  int update_position = 0;
  std::vector<IpNeighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  std::mt19937 rng(rand());
  GenRandom(rng, init_ids.data(), L, (unsigned)nd_);
  // assert(eps_.size() < L);
  for(unsigned i=0; i<eps_.size(); i++){
    init_ids[i] = eps_[i];
  }
  
  DistanceInnerProduct dist_inner;
  for (unsigned i = 0; i < L; i++) {
    unsigned id = init_ids[i];
    float ip = dist_inner.compare(data_ + dimension_ * id, query,
                                    dimension_);
    retset[i] = IpNeighbor(id, ip, true);
    flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  
  int k = 0;
  while (k < (int)L) {
    int nk = L;
    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;
      int search_pos = final_graph_[n].size();
      for (unsigned m = 0; m < search_pos; ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id]) continue;
        flags[id] = 1;        
        float dist = dist_inner.compare(query, data_ + dimension_ * id,
                                        dimension_); 
        dis_cal += 1;
        if (dist <= retset[L - 1].distance) continue;
        IpNeighbor nn(id, dist, true);
        int r = InsertIntoIpPool(retset.data(), L, nn);
        if (r < nk) {
          nk = r;
        }
      }
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
  return {dis_cal};
}



void IndexMips::DFS(boost::dynamic_bitset<> &flag,
                   std::vector<std::pair<unsigned, unsigned>> &edges,
                   unsigned root, unsigned &cnt) {
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if (!flag[root]) cnt++;
  flag[root] = true;
  while (!s.empty()) {
    unsigned next = nd_ + 1;
    for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
      if (flag[final_graph_[tmp][i]] == false) {
        next = final_graph_[tmp][i];
        break;
      }
    }
    if (next == (nd_ + 1)) {
      unsigned head = s.top();
      s.pop();
      if (s.empty()) break;
      tmp = s.top();
      unsigned tail = tmp;
      if (check_edge(head, tail)) {
        edges.push_back(std::make_pair(head, tail));
      }
      continue;
    }
    tmp = next;
    flag[tmp] = true;
    s.push(tmp);
    cnt++;
  }
}




int IndexMips::check_connected_component(){
  int cnt = 0;
  boost::dynamic_bitset<> flags{nd_, 0};
  for (unsigned i = 0; i < nd_; i++) {
    if (flags[i] == false) {
      cnt++;
      unsigned root = i;
      std::queue<unsigned> myqueue;
      myqueue.push(root);
      while (!myqueue.empty()) {
        unsigned q_front = myqueue.front();
        myqueue.pop();
        if (flags[q_front]) continue;
        flags[q_front] = true;
        for (unsigned j = 0; j < final_graph_[q_front].size(); j++) {
          unsigned child = final_graph_[q_front][j];
          if (flags[child]) continue;
          myqueue.push(child);
        }
      }

    }
  }
  return cnt;
}

void IndexMips::findroot(boost::dynamic_bitset<> &flag, unsigned &root,
                        const Parameters &parameter) {
  unsigned id = nd_;
  for (unsigned i = 0; i < nd_; i++) {
    if (flag[i] == false) {
      id = i;
      break;
    }
  }

  if (id == nd_) return;  // No Unlinked Node

  std::vector<Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  bool found = false;
  for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
      // std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = true;
      break;
    }
  }
  if (!found) {
    for (int retry = 0; retry < 1000; ++retry) {
      unsigned rid = rand() % nd_;
      if (flag[rid]) {
        root = rid;
        break;
      }
    }
  }
  final_graph_[root].push_back(id);
}

bool IndexMips::check_edge(unsigned h, unsigned t) {
  bool flag = true;
  for (unsigned i = 0; i < final_graph_[h].size(); i++) {
    if (t == final_graph_[h][i]) flag = false;
  }
  return flag;
}

void IndexMips::strong_connect(const Parameters &parameter) {
  unsigned n_try = parameter.Get<unsigned>("n_try");
  std::vector<std::pair<unsigned, unsigned>> edges_all;
  std::mutex edge_lock;

#pragma omp parallel for
  for (unsigned nt = 0; nt < n_try; nt++) {
    unsigned root = rand() % nd_;
    boost::dynamic_bitset<> flags{nd_, 0};
    unsigned unlinked_cnt = 0;
    std::vector<std::pair<unsigned, unsigned>> edges;

    while (unlinked_cnt < nd_) {
      DFS(flags, edges, root, unlinked_cnt);
      if (unlinked_cnt >= nd_) break;
      findroot(flags, root, parameter);
    }

    LockGuard guard(edge_lock);

    for (unsigned i = 0; i < edges.size(); i++) {
      edges_all.push_back(edges[i]);
    }
  }
  unsigned ecnt = 0;
  for (unsigned e = 0; e < edges_all.size(); e++) {
    unsigned start = edges_all[e].first;
    unsigned end = edges_all[e].second;
    unsigned flag = 1;
    for (unsigned j = 0; j < final_graph_[start].size(); j++) {
      if (end == final_graph_[start][j]) {
        flag = 0;
      }
    }
    if (flag) {
      final_graph_[start].push_back(end);
      ecnt++;
    }
  }
  for (size_t i = 0; i < nd_; ++i) {
    if (final_graph_[i].size() > width) {
      width = final_graph_[i].size();
    }
  }
}

void IndexMips::DFS_expand(const Parameters &parameter) {
  unsigned n_try = parameter.Get<unsigned>("n_try");
  unsigned range = parameter.Get<unsigned>("R");

  std::vector<unsigned> ids(nd_);
  for(unsigned i=0; i<nd_; i++){
    ids[i]=i;
  }
  std::random_shuffle(ids.begin(), ids.end());
  for(unsigned i=0; i<n_try; i++){
    eps_.push_back(ids[i]);
  }
#pragma omp parallel for
  for(unsigned i=0; i<n_try; i++){
    unsigned rootid = eps_[i];
    boost::dynamic_bitset<> flags{nd_, 0};
    std::queue<unsigned> myqueue;
    myqueue.push(rootid);
    flags[rootid]=true;
    std::vector<unsigned> uncheck_set(1);
  
    while(uncheck_set.size() >0){
      while(!myqueue.empty()){
        unsigned q_front=myqueue.front();
        myqueue.pop();

        for(unsigned j=0; j<final_graph_[q_front].size(); j++){
          unsigned child = final_graph_[q_front][j];
          if(flags[child])continue;
          flags[child] = true;
          myqueue.push(child);

        }
      }

      uncheck_set.clear();
      for(unsigned j=0; j<nd_; j++){
        if(flags[j]) continue;
        uncheck_set.push_back(j);
      }
      if(uncheck_set.size()>0){
        for(unsigned j=0; j<nd_; j++){
          if(flags[j] && final_graph_[j].size()< (range + 1)){
            final_graph_[j].push_back(uncheck_set[0]);
            break;
          }
        }
        myqueue.push(uncheck_set[0]);
        flags[uncheck_set[0]]=true;
      }
    }
  }
}
};
