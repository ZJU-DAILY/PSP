#pragma once

#include <pthread.h>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <random>
#include <vector>

#include "util.h"

namespace efanna2e {

struct Neighbor {
  unsigned id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(unsigned id, float distance, bool f)
      : id{id}, distance{distance}, flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct IpNeighbor {
  unsigned id;
  float distance;
  bool flag;

  IpNeighbor() = default;
  IpNeighbor(unsigned id, float distance, bool f)
      : id{id}, distance{distance}, flag(f) {}

  inline bool operator<(const IpNeighbor &other) const {
    return distance > other.distance;
  }
};

typedef std::lock_guard<std::mutex> LockGuard;
struct nhood {
  // std::mutex lock;
  pthread_mutex_t lock;
  std::vector<Neighbor> pool;
  unsigned M;

  std::vector<unsigned> nn_old;
  std::vector<unsigned> nn_new;
  std::vector<unsigned> rnn_old;
  std::vector<unsigned> rnn_new;

  nhood() {}
  nhood(unsigned l, unsigned s, std::mt19937 &rng, unsigned N) {
    M = s;
    nn_new.resize(s * 2);
    GenRandom(rng, &nn_new[0], (unsigned)nn_new.size(), N);
    nn_new.reserve(s * 2);
    pool.reserve(l);
    pthread_mutex_init(&lock, NULL);
  }

  nhood(const nhood &other) {
    M = other.M;
    std::copy(other.nn_new.begin(), other.nn_new.end(),
              std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
  }
  void insert(unsigned id, float dist) {
    // LockGuard guard(lock);
    pthread_mutex_lock(&lock);
    if (dist > pool.front().distance) {
      pthread_mutex_unlock(&lock);
      return;
    }
    for (unsigned i = 0; i < pool.size(); i++) {
      if (id == pool[i].id) {
        pthread_mutex_unlock(&lock);
        return;
      }
    }
    if (pool.size() < pool.capacity()) {
      pool.push_back(Neighbor(id, dist, true));
      std::push_heap(pool.begin(), pool.end());
    } else {
      std::pop_heap(pool.begin(), pool.end());
      pool[pool.size() - 1] = Neighbor(id, dist, true);
      std::push_heap(pool.begin(), pool.end());
    }
    pthread_mutex_unlock(&lock);
  }

  template <typename C>
  void join(C callback) const {
    for (unsigned const i : nn_new) {
      for (unsigned const j : nn_new) {
        if (i < j) {
          callback(i, j);
        }
      }
      for (unsigned j : nn_old) {
        callback(i, j);
      }
    }
  }
};

struct LockNeighbor {
  // std::mutex lock;
  pthread_mutex_t lock;
  std::vector<Neighbor> pool;
};

struct SimpleNeighbor {
  unsigned id;
  float distance;

  SimpleNeighbor() = default;
  SimpleNeighbor(unsigned id_, float distance_)
      : id(id_), distance(distance_) {}

  inline bool operator<(const SimpleNeighbor &other) const {
    return distance < other.distance;
  }
};

struct SimpleIpNeighbor {
  unsigned id;
  float distance;

  SimpleIpNeighbor() = default;
  SimpleIpNeighbor(unsigned id_, float distance_)
      : id(id_), distance(distance_) {}

  inline bool operator<(const SimpleIpNeighbor &other) const {
    return distance > other.distance;
  }
};

struct SimpleNeighbors {
  std::vector<SimpleNeighbor> pool;
};

struct SimpleIpNeighbors {
  std::vector<SimpleIpNeighbor> pool;
};

static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance)
      right = mid;
    else
      left = mid;
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
  memmove((char *)&addr[right + 1], &addr[right],
          (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

static inline int InsertIntoIpPool(IpNeighbor *addr, unsigned K, IpNeighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance < nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], K * sizeof(IpNeighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance > nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance < nn.distance)
      right = mid;
    else
      left = mid;
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) break;
    if (addr[left].id == nn.id) return K + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
  memmove((char *)&addr[right + 1], &addr[right],
          (K - right) * sizeof(IpNeighbor));
  addr[right] = nn;
  return right;
}

} 