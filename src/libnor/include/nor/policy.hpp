
#ifndef NOR_POLICY_HPP
#define NOR_POLICY_HPP

#include "nor/concepts.hpp"

namespace nor {

template < typename Map >
requires requires(Map m)
{
   std::is_default_constructible_v< Map >;
   Map::key_type;
   Map::mapped_type;
}
class TabularPolicy {

   using key_type = typename Map::key_type;
   using mapped_type = typename Map::mapped_type;

   TabularPolicy() = default;

   auto& operator[](const key_type& key) { return m_table[key]; }
   auto& operator[](const key_type& key) const { return m_table.at(key); }

  private:
   Map m_table;
};

}  // namespace nor

#endif  // NOR_POLICY_HPP
