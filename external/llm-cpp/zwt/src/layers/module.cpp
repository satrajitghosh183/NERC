#include "zwt/layers/module.hpp"

namespace zwt {

void step_begin() {
  // Reset the activation arenas so the next forward starts from offset zero.
  auto& cpu = activation_arena(Device::cpu());
  if (auto* a = dynamic_cast<ArenaAllocator*>(&cpu))  a->reset();
#ifdef USE_CUDA
  auto& cuda = activation_arena(Device::cuda());
  if (auto* a = dynamic_cast<ArenaAllocator*>(&cuda)) a->reset();
#endif
}

}  // namespace zwt
