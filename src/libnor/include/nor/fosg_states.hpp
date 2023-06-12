
#ifndef NOR_FOSG_STATES_HPP
#define NOR_FOSG_STATES_HPP

#include <range/v3/all.hpp>
#include <string>
#include <vector>

#include "nor/game_defs.hpp"
#include "nor/rm/tag.hpp"
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
      return _build_string([this](std::string& s) {
         for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
            s += std::get< 0 >(observation) + "\n";
            s += std::get< 1 >(observation) + "\n";
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
            s += std::get< 0 >(observation);
            s += "\n";
            s += "prv_";
            s += round_str;
            s += ": ";
            s += std::get< 1 >(observation);
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

   auto to_string(std::string_view delim = "\n", std::string_view sep = ",") const
      requires std::is_same_v< observation_type, std::string >
   {
      constexpr size_t avg_string_size_expectation = 500;
      std::string s{};
      s.reserve(size() * avg_string_size_expectation);
      for(const auto& [pos, observation] : ranges::views::enumerate(m_history)) {
         s += "{";
         s += std::get< 0 >(observation);
         s += sep;
         s += std::get< 1 >(observation);
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
         constexpr auto string_hasher = std::hash< std::string >{};
         const auto& latest_entry = m_history.back();
         common::hash_combine(m_hash_cache, string_hasher(std::get< 0 >(latest_entry)));
         common::hash_combine(m_hash_cache, string_hasher(std::get< 1 >(latest_entry)));
      } else {
         m_hash_cache = static_cast< derived_type* >(this)->_hash_impl();
      }
   }
};
template < template < typename > class DerivedWrapper, typename Type >
struct BaseWrapper {
   using derived_wrapper_type = DerivedWrapper< Type >;
   using underlying_type = Type;
   static constexpr bool is_polymorphic = std::is_polymorphic_v< underlying_type >;
   using holder_type = std::
      conditional_t< is_polymorphic, sptr< underlying_type >, underlying_type >;

  protected:
   /// internal constructor to actually allocate the member
   template < typename... Ts >
   BaseWrapper(tag::internal_construct, Ts&&... args)
       : m_member(_init_member(std::forward< Ts >(args)...))
   {
   }

  public:
   template < typename T1, typename... Ts >
      requires common::is_none_v<
         std::decay_t< T1 >,  // decay_t to avoid checking each ref-case individually
         tag::internal_construct,  // don't recurse back from internal constructors or self
         BaseWrapper  // don't steal the copy/move constructor calls (std::remove_cvref ensures
                      // both)
         >
   explicit BaseWrapper(T1&& t1, Ts&&... args)
       : BaseWrapper(tag::internal_construct{}, std::forward< T1 >(t1), std::forward< Ts >(args)...)
   {
   }

   BaseWrapper(sptr< underlying_type > ptr)
      requires is_polymorphic
       : m_member(std::move(ptr))
   {
   }

   BaseWrapper()
      requires std::is_default_constructible_v< underlying_type >
       : BaseWrapper(tag::internal_construct{})
   {
   }
   /// implicit converion operators so that the contained type will be passable to functions that
   /// use the underlying type (similar to reference_wrapper's behaviour)
   constexpr operator const underlying_type&() const noexcept { return get(); }
   constexpr operator underlying_type&() noexcept { return get(); }

   [[nodiscard]] const underlying_type& get() const
   {
      if constexpr(is_polymorphic) {
         return *m_member;
      } else {
         return m_member;
      }
   }
   [[nodiscard]] underlying_type& get()
   {
      // clang-tidy complains, but since the member function calling it is non-const, the object
      // itself is non-const. Thus, casting away the const is fine.
      return const_cast< underlying_type& >(std::as_const(*this).get());
   }

   bool operator==(const BaseWrapper& other) const
      requires(std::equality_comparable< underlying_type >)
   {
      return other.get() == get();
   }

   [[nodiscard]] derived_wrapper_type copy() const
   {
      // copy_tag to avoid the implicit copy constructor
      return derived_wrapper_type{tag::internal_construct{}, get()};
   }

   [[nodiscard]] std::string to_string() const
      requires requires(underlying_type obj) { obj.to_string(); }
   {
      return get().to_string();
   }

   const BaseWrapper* operator->() const { return &get(); }
   BaseWrapper* operator->() { return &get(); }

  private:
   holder_type m_member;

   template < typename FirstArg, typename... Args >
   [[nodiscard]] holder_type _init_member(FirstArg&& first_arg, Args&&... args) const
   {
      constexpr auto _init_member_impl = []< typename... Ts >(Ts&&... ts) {
         if constexpr(derived_wrapper_type::is_polymorphic) {
            return std::make_shared< underlying_type >(std::forward< Ts >(ts)...);
         } else {
            return underlying_type{std::forward< Ts >(ts)...};
         }
      };
      if constexpr(std::is_same_v< std::decay_t< FirstArg >, tag::internal_construct >) {
         return _init_member_impl(std::forward< Args >(args)...);
      } else {
         return _init_member_impl(
            std::forward< FirstArg >(first_arg), std::forward< Args >(args)...
         );
      }
   }
};

template < typename Action >
struct ActionWrapper: public BaseWrapper< ActionWrapper, Action > {
   using base = BaseWrapper< ActionWrapper, Action >;
   using base::base;
};

template < typename ChanceOutcome >
struct ChanceOutcomeWrapper: public BaseWrapper< ChanceOutcomeWrapper, ChanceOutcome > {
   using base = BaseWrapper< ChanceOutcomeWrapper, ChanceOutcome >;
   using base::base;
};

template < typename Observation >
struct ObservationWrapper: public BaseWrapper< ObservationWrapper, Observation > {
   using base = BaseWrapper< ObservationWrapper, Observation >;
   using base::base;
};

template < typename Worldstate >
struct WorldstateWrapper: public BaseWrapper< WorldstateWrapper, Worldstate > {
   using base = BaseWrapper< WorldstateWrapper, Worldstate >;
   using base::base;
};

template < typename Infostate >
struct InfostateWrapper: public BaseWrapper< InfostateWrapper, Infostate > {
   using base = BaseWrapper< InfostateWrapper, Infostate >;
   using base::base;
   using base::get;
   using observation_type = typename fosg_auto_traits< Infostate >::observation_type;

   [[nodiscard]] size_t size() const { return get().size(); }

   const auto& update(const observation_type& public_obs, const observation_type& private_obs)
   {
      return get().update(public_obs, private_obs);
   }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }

   [[nodiscard]] Player player() const { return get().player(); }
};

template < typename Publicstate >
struct PublicstateWrapper: public BaseWrapper< PublicstateWrapper, Publicstate > {
   using base = BaseWrapper< PublicstateWrapper, Publicstate >;
   using base::base;
   using base::get;
   using observation_type = typename fosg_auto_traits< Publicstate >::observation_type;

   [[nodiscard]] size_t size() const { return get().size(); }

   const auto& update(const observation_type& public_obs, const observation_type& private_obs)
   {
      return get().update(public_obs, private_obs);
   }

   auto& operator[](auto index) { return get()[size_t(index)]; }

   const auto& operator[](auto index) const { return get()[size_t(index)]; }
};

}  // namespace nor
namespace std {

template < typename T >
   requires common::logical_or_v<
      common::is_specialization< std::remove_cvref_t< T >, nor::ActionWrapper >,
      common::is_specialization< std::remove_cvref_t< T >, nor::ObservationWrapper >,
      common::is_specialization< std::remove_cvref_t< T >, nor::InfostateWrapper >,
      common::is_specialization< std::remove_cvref_t< T >, nor::PublicstateWrapper > >
struct hash< T > {
   size_t operator()(const T& t) const
   {
      constexpr auto underlying_hasher = std::hash< std::conditional_t<
         std::remove_cvref_t< T >::is_polymorphic,
         typename std::remove_cvref_t< T >::underlying_type::element_type,
         typename std::remove_cvref_t< T >::underlying_type > >{};

      return underlying_hasher(t.get());
   }
};

}  // namespace std

#endif  // NOR_FOSG_STATES_HPP
