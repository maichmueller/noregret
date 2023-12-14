
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

   auto& operator[](std::convertible_to< size_t > auto index) const
   {
      return m_history[size_t(index)];
   }

   auto& latest() const { return m_history.back(); }

   [[nodiscard]] auto& history() const { return m_history; }
   [[nodiscard]] size_t size() const { return m_history.size(); }

   void update(const observation_type& public_obs)
   {
      m_history.emplace_back(public_obs);
      _hash();
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      return _build_string([this](std::string& str) {
         for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
            const auto& [pub_obs, priv_obs] = observation;
            fmt::format_to(std::back_inserter(str), "{}\n{}", pub_obs, priv_obs);
         };
      });
   }

   auto to_pretty_string() const
      requires std::is_same_v< observation_type, std::string >
   {
      using namespace fmt::literals;
      return _build_string([this](std::string& str) {
         for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
            const auto& [pub_obs, priv_obs] = observation;
            fmt::format_to(
               std::back_inserter(str),
               "pub_{round}: {}\n"
               "prv_{round}: {}\n",
               "round"_a = fmt::format("obs_{}", pos),
               pub_obs,
               priv_obs
            );
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
   std::vector< observation_type > m_history{};
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
      std::string str{};
      str.reserve(size() * avg_string_size_expectation);
      // let the builder do its job
      builder(str);
      // reduce size to only eat as much memory as needed
      str.shrink_to_fit();
      return str;
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
      m_history.emplace_back(public_obs, private_obs);
      _hash();
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string(std::string_view delim = "\n", std::string_view sep = ",") const
      requires std::is_same_v< observation_type, std::string >
   {
      using namespace fmt::literals;
      constexpr auto opener = "{";
      constexpr auto closer = "}";
      constexpr size_t avg_string_size_expectation = 500;
      std::string str{};
      str.reserve(size() * avg_string_size_expectation);
      for(const auto& observation : m_history | ranges::views::drop_last(1)) {
         const auto& [pub_obs, priv_obs] = observation;
         fmt::format_to(
            std::back_inserter(str),
            "{opener}{pub_obs}{sep}{priv_obs}{closer}{delim}",
            "opener"_a = opener,
            "closer"_a = closer,
            "pub_obs"_a = pub_obs,
            "priv_obs"_a = priv_obs,
            "sep"_a = sep,
            "delim"_a = delim
         );
      }
      if(m_history.size() > 1) {
         // the very last observation only
         const auto& [pub_obs, priv_obs] = m_history.back();
         fmt::format_to(
            std::back_inserter(str),
            "{opener}{pub_obs}{sep}{priv_obs}{closer}",
            "opener"_a = opener,
            "closer"_a = closer,
            "pub_obs"_a = pub_obs,
            "priv_obs"_a = priv_obs,
            "sep"_a = sep
         );
      }
      str.shrink_to_fit();
      return str;
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
            return public_obs_1 == public_obs_2 and private_obs_1 == private_obs_2;
         }
      );
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
         constexpr auto string_hasher = std::hash< std::string >{};
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
