
#ifndef NOR_FOSG_STATES_HPP
#define NOR_FOSG_STATES_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/utils/utils.hpp"

namespace nor {

/**
 * @brief A default Publicstate type building on vectors of observations for environments to use as
 * plugin.
 *
 * We are using CRTP to allow for the derived class to bring a hash function with it that
 * corresponds to the provided observation type. E.g. if a std::string is the observation then the
 * base class should provide a private _hash function that uses the strings to compute an
 * appropriate hash.
 *
 * @tparam Derived
 * @tparam Observation
 */
template < typename Derived, concepts::observation Observation >
class DefaultPublicstate {
  public:
   using derived_type = Derived;
   using observation_type = Observation;

   DefaultPublicstate() = default;

   auto& operator[](std::convertible_to< size_t > auto index) { return m_history[size_t(index)]; }

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   auto& update(observation_type public_obs)
   {
      auto& ret_val = m_history.emplace_back(std::move(public_obs));
      _hash();
      return ret_val;
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      std::string s;
      for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
         s += std::get< 0 >(observation) + "\n";
         s += std::get< 1 >(observation) + "\n";
      }
      return s;
   }

   auto to_pretty_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      std::string s;
      for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
         auto round_str = std::string("obs_") + std::to_string(pos);
         s += "pub_" + round_str + ": " + std::get< 0 >(observation) + "\n";
         s += "prv_" + round_str + ": " + std::get< 1 >(observation) + "\n";
      }
      return s;
   }

   bool operator==(const DefaultPublicstate& other) const
   {
      if(size() != other.size()) {
         return false;
      }
      return ranges::all_of(ranges::views::zip(m_history, other.history()), [](const auto& pair) {
         return std::get< 0 >(pair) == std::get< 1 >(pair);
      });
   }
   inline bool operator!=(const DefaultPublicstate& other) const { return not (*this == other); }

  private:
   /// the private_history (action trajectory) container of the state.
   /// Each entry is an observation of a state followed by an action.
   std::vector< Observation > m_history{};
   /// the cache of the current hash value of the public state
   size_t m_hash_cache{0};

   void _hash()
   {
      if constexpr(std::is_same_v< observation_type, std::string >) {
         common::hash_combine(m_hash_cache, std::hash< std::string >{}(m_history.back()));
      } else {
         m_hash_cache = static_cast< derived_type* >(this)->_hash_impl();
      }
   }
};

template < typename Derived, concepts::observation Observation >
class DefaultInfostate {
  public:
   using derived_type = Derived;
   using observation_type = Observation;

   DefaultInfostate(Player player) : m_player(player){};

   auto& operator[](std::convertible_to< size_t > auto index) { return m_history[size_t(index)]; }

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   auto& update(observation_type public_obs, observation_type private_obs)
   {
      auto& ret_val = m_history.emplace_back(std::pair{
         std::move(public_obs), std::move(private_obs)});
      _hash();
      return ret_val;
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string(std::string delim = "\n") const
      requires std::is_same_v< observation_type, std::string >
   {
      std::string s;
      for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
         s += std::get< 0 >(observation) + delim;
         s += std::get< 1 >(observation) + delim;
      }
      return s;
   }

   bool operator==(const DefaultInfostate& other) const
   {
      if(size() != other.size()) {
         return false;
      }
      return ranges::all_of(ranges::views::zip(m_history, other.history()), [](const auto& pair) {
         return std::get< 0 >(pair) == std::get< 1 >(pair);
      });
   }
   inline bool operator!=(const DefaultInfostate& other) const { return not (*this == other); }

   [[nodiscard]] auto player() const { return m_player; }

  private:
   Player m_player;
   /// the private_history (action trajectory) container of the state.
   /// Each entry is an observation of a state followed by an action.
   std::vector< std::pair< Observation, Observation > > m_history{};
   /// the cache of the current hash value of the public state
   size_t m_hash_cache{0};

   void _hash()
   {
      if constexpr(std::is_same_v< observation_type, std::string >) {
         auto string_hasher = std::hash< std::string >{};
         const auto& latest_entry = m_history.back();
         common::hash_combine(m_hash_cache, string_hasher(std::get< 0 >(latest_entry)));
         common::hash_combine(m_hash_cache, string_hasher(std::get< 1 >(latest_entry)));
      } else {
         m_hash_cache = static_cast< derived_type* >(this)->_hash_impl();
      }
   }
};

}  // namespace nor

#endif  // NOR_FOSG_STATES_HPP
