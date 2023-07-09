
#ifndef NOR_FOSG_STATES_HPP
#define NOR_FOSG_STATES_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/holder.hpp"
#include "nor/tag.hpp"
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

   auto& operator[](std::convertible_to< size_t > auto index) const
   {
      return m_history[size_t(index)];
   }

   auto& latest() const { return m_history.back(); }

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   void update(const observation_type& public_obs)
   {
      m_history.emplace_back(ObservationHolder< observation_type >{public_obs});
      _hash();
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      return _build_string([this](std::string& s) {
         for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
            s += std::get< 0 >(observation.get()) + "\n";
            s += std::get< 1 >(observation.get()) + "\n";
         };
      });
   }

   auto to_pretty_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      return _build_string([this](std::string& s) {
         for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
            auto round_str = "obs_" + std::to_string(pos);
            s += "pub_";
            s += round_str;
            s += ": ";
            s += std::get< 0 >(observation).get();
            s += "\n";
            s += "prv_";
            s += round_str;
            s += ": ";
            s += std::get< 1 >(observation).get();
            s += "\n";
         }
      });
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
   std::vector< ObservationHolder< observation_type > > m_history{};
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

   template < std::invocable< std::string& > BuildFunctor >
   auto _build_string(BuildFunctor builder) const
   {
      constexpr size_t avg_string_size_expectation = 500;
      std::string s{};
      s.reserve(size() * avg_string_size_expectation);
      // let the builder do its job
      builder(s);
      // reduce size to only eat as much memory as needed
      s.shrink_to_fit();
      return s;
   }
};

template < typename Derived, concepts::observation Observation >
class DefaultInfostate {
  public:
   using derived_type = Derived;
   using observation_type = Observation;

   DefaultInfostate(Player player)
       : m_player(player), m_hash_cache(std::hash< int >{}(static_cast< int >(player))){};

   auto& operator[](std::convertible_to< size_t > auto index) const
   {
      return m_history[size_t(index)];
   }

   auto& latest() const { return m_history.back(); }

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   void update(const observation_type& public_obs, const observation_type& private_obs)
   {
      m_history.emplace_back(
         ObservationHolder< observation_type >{public_obs},
         ObservationHolder< observation_type >{private_obs}
      );
      _hash();
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string(std::string_view delim = "\n", std::string_view sep = ",") const
      requires std::is_same_v< observation_type, std::string >
   {
      constexpr size_t avg_string_size_expectation = 500;
      std::string s{};
      s.reserve(size() * avg_string_size_expectation);
      for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
         s += "{";
         s += std::get< 0 >(observation).get();
         s += sep;
         s += std::get< 1 >(observation).get();
         s += "}";
         s += delim;
      }
      s.shrink_to_fit();
      return s;
   }

   bool operator==(const DefaultInfostate& other) const
   {
      if(size() != other.size() or m_player != other.player()) {
         return false;
      }
      return ranges::all_of(
         ranges::views::zip(m_history, other.history()),
         [](const auto& pair_of_pairs) {
            const auto& [pair1, pair2] = pair_of_pairs;
            const auto& [public_obs_1, private_obs_1] = pair1;
            const auto& [public_obs_2, private_obs_2] = pair2;
            return public_obs_1.equals(public_obs_2) and private_obs_1.equals(private_obs_2);
         }
      );
   }
   inline bool operator!=(const DefaultInfostate& other) const { return not (*this == other); }

   [[nodiscard]] auto player() const { return m_player; }

  private:
   Player m_player;
   /// the private_history (action trajectory) container of the state.
   /// Each entry is an observation of a state followed by an action.
   std::vector< std::pair< ObservationHolder< Observation >, ObservationHolder< Observation > > >
      m_history{};
   /// the cache of the current hash value of the public state
   size_t m_hash_cache{0};

   void _hash()
   {
      if constexpr(std::is_same_v< observation_type, std::string >) {
         constexpr auto string_hasher = std::hash< std::string >{};
         const auto& latest_entry = m_history.back();
         common::hash_combine(m_hash_cache, string_hasher(std::get< 0 >(latest_entry).get()));
         common::hash_combine(m_hash_cache, string_hasher(std::get< 1 >(latest_entry).get()));
      } else {
         m_hash_cache = static_cast< derived_type* >(this)->_hash_impl();
      }
   }
};

}  // namespace nor

#endif  // NOR_FOSG_STATES_HPP
