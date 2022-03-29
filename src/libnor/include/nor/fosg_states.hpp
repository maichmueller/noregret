
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

   template < typename... Args >
   auto& append(Args&&... args)
   {
      auto& ret_val = m_history.emplace_back(std::forward< Args >(args)...);
      m_hash_cache = _hash();
      return ret_val;
   }

   [[nodiscard]] size_t hash() const { return m_hash_cache; }

   auto to_string() const requires std::is_same_v< observation_type, std::string >
   {
      std::stringstream ss;
      size_t pos = 0;
      for(const auto& [action, observation] : m_history) {
         ss << "a_" << pos << ":[" << action << "]\n";
         ss << "obs_" << pos << ":[" << observation << "]\n";
         ss << "-----\n";
         pos++;
      }
      return ss.str();
   };

   bool operator==(const DefaultPublicstate& other) const
   {
      if(size() != other.size()) {
         return false;
      }
      auto zip_view = ranges::views::zip(m_history, other.history());
      return std::all_of(zip_view.begin(), zip_view.end(), [](const auto& tuple) {
         const auto& [this_hist_elem, other_hist_elem] = tuple;
         return this_hist_elem.first == other_hist_elem.first
                and this_hist_elem.second == other_hist_elem.second;
      });
   }
   inline bool operator!=(const DefaultPublicstate& other) const { return not (*this == other); }

  private:
   /// the history (action trajectory) of the state.
   /// Each entry is a pair of action observation (first) and state observation (second)
   std::vector< std::pair< Observation, Observation > > m_history{};
   size_t m_hash_cache{0};

   size_t _hash()
   {
      if constexpr(std::is_same_v< observation_type, std::string >) {
         if(requires(derived_type d) {
               // clang-format off
                      { d.to_string() } -> std::same_as< std::string >;
               // clang-format on
            }) {
            return std::hash< std::string >{}(static_cast< derived_type* >(this)->to_string());
         } else {
            return std::hash< std::string >{}(to_string());
         }
      } else {
         return static_cast< derived_type* >(this)->_hash_impl();
      }
   }
};

template < typename Derived, concepts::observation Observation >
class DefaultInfostate: public DefaultPublicstate< Derived, Observation > {
  public:
   using base = DefaultPublicstate< Derived, Observation >;
   using observation_type = typename base::observation_type;

   DefaultInfostate(Player player) : base(), m_player(player) {}

   [[nodiscard]] Player player() const { return m_player; }

   auto to_string() const requires std::is_same_v< observation_type, std::string >
   {
      std::stringstream ss;
      ss << m_player << "\n" << base::to_string();
      return ss.str();
   };

  private: Player m_player;
};

}  // namespace nor

#endif  // NOR_FOSG_STATES_HPP
